"""Одинаковые задачи для трех настоящих локальных моделей, без подмены ответов."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from multi_agent.llm import ROOT, ModelRouter
from multi_agent.schemas import Draft, Plan, Verdict


def cases():
    catalog = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
    tasks = [
        {"id": "route_genres", "role": "planner", "payload": {
            "request": "Посоветуй три комедии без ужасов", "catalog_genres": catalog,
            "profile": {}}, "expected": {"tool": "recommend", "method": "item_cf", "genres": ["Comedy"], "avoid": ["Horror"], "k": 3}},
        {"id": "route_history", "role": "planner", "payload": {
            "request": "Покажи мою историю оценок", "catalog_genres": catalog,
            "profile": {}}, "expected": {"tool": "history", "method": "item_cf"}},
        {"id": "route_search", "role": "planner", "payload": {
            "request": "Найди фильм Toy Story", "catalog_genres": catalog,
            "profile": {}}, "expected": {"tool": "search", "query": "Toy Story"}},
        {"id": "route_compare", "role": "planner", "payload": {
            "request": "Сравни рекомендации item_cf и LightGCN", "catalog_genres": catalog,
            "profile": {}}, "expected": {"tool": "compare"}},
    ]
    for name, k, items in (
        ("write_comedy", 2, [{"item_id": 12, "title": "Movie A", "genres": ["Comedy"]}, {"item_id": 34, "title": "Movie B", "genres": ["Comedy", "Drama"]}]),
        ("write_scifi", 1, [{"item_id": 56, "title": "Movie C", "genres": ["Sci-Fi"]}, {"item_id": 78, "title": "Movie D", "genres": ["Sci-Fi", "Action"]}]),
    ):
        tasks.append({"id": name, "role": "writer", "payload": {
            "request": f"Выбери первые {k} фильма", "k": k, "profile": {},
            "evidence": {"items": items, "method": "item_cf"}, "feedback": []},
            "expected": {"ids": [row["item_id"] for row in items[:k]]}})
    evidence = {"items": [{"item_id": 12, "title": "Movie A", "genres": ["Comedy"]},
                          {"item_id": 34, "title": "Movie B", "genres": ["Drama"]}],
                "method": "item_cf"}
    for name, choices, approved in (
        ("judge_valid", [{"item_id": 12, "reason": "Комедия из списка рекомендаций item_cf."}], True),
        ("judge_unknown_id", [{"item_id": 999, "reason": "Комедия."}], False),
        ("judge_invented_plot", [{"item_id": 12, "reason": "История космонавта, который спасает Марс, с Томом Хэнксом."}], False),
        ("judge_duplicate", [{"item_id": 12, "reason": "Комедия."}, {"item_id": 12, "reason": "Комедия."}], False),
    ):
        tasks.append({"id": name, "role": "validator", "payload": {
            "request": "Посоветуй фильм из каталога", "k": 2 if name == "judge_duplicate" else 1,
            "draft": {"choices": choices}, "evidence": evidence},
            "expected": {"approved": approved}})
    return tasks


def check_answer(case, data):
    role = case["role"]
    if role == "planner":
        for key, value in case["expected"].items():
            actual = data.get(key)
            if key == "query":
                if actual.lower().strip() != value.lower():
                    return False
            elif actual != value:
                return False
        return True
    if role == "validator":
        return data["approved"] == case["expected"]["approved"] and bool(data["issues"]) != data["approved"]
    ids = [row["item_id"] for row in data["choices"]]
    # Проверка содержания причин не заменяет ручной просмотр raw_content.
    return ids == case["expected"]["ids"] and all(row["reason"].strip() for row in data["choices"])


def plot_summary(models, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["name"] + "\n" + row["quantization"] for row in models]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    values = [row["aggregate"]["task_accuracy"] * 100 for row in models]
    bars = axes[0].bar(names, values, color="#597b9b")
    axes[0].set(ylabel="Верные задачи, %", ylim=(0, 110), title="10 одинаковых задач")
    axes[0].bar_label(bars, fmt="%.0f%%", padding=3)
    times = [row["aggregate"]["mean_inference_s"] or 0 for row in models]
    bars = axes[1].bar(names, times, color="#868f79")
    axes[1].set(ylabel="Секунды", title="Генерация без загрузки модели")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("llama.cpp, CPU 4 потока, контекст 4096, JSON Schema", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "multi_agent/results/model_comparison.json")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schemas = {"planner": Plan, "writer": Draft, "validator": Verdict}
    prompts = {role: (ROOT / f"multi_agent/prompts/{role}.md").read_text() for role in schemas}
    config = json.loads((ROOT / "multi_agent/models.json").read_text())
    selected = args.models or list(config["models"])
    for key in selected:
        model = config["models"][key]
        weights = ROOT / ".cache/models" / model["filename"]
        if not weights.exists() or weights.stat().st_size != model["size_bytes"]:
            raise RuntimeError(f"Сначала полностью скачайте веса {model['name']}")
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "engine": config["engine"],
        "inference": config["inference"],
    }
    engine_path = Path(os.environ.get("LLAMA_SERVER", str(ROOT / config["engine"]["path"])))
    environment["engine_version"] = subprocess.run([str(engine_path), "--version"], capture_output=True, text=True, check=True).stderr.strip()
    result = {
        "protocol": {
            "cases_per_model": len(cases()), "identical_cases": True,
            "schema_constrained_decoding": True, "temperature": 0, "seed": 42,
            "note": "Маленький функциональный набор, не общий рейтинг LLM. JSON grammar помогает формату; смысл проверяется отдельно. Writer score проверяет ID/порядок/непустую причину, содержание причин требует просмотра сырых ответов.",
            "timing": "inference_s — HTTP генерация; load_s — запуск/загрузка; latency_s — их сумма. Холодный запуск учтен отдельно от среднего inference.",
            "timing_note": "cold_load_s — первый запуск процесса модели в серии. Файловый кеш ОС не очищался; температура ноутбука и фоновые процессы не фиксировались. Это один локальный прогон, не аппаратный бенчмарк.",
            "prompts": prompts,
            "prompt_sha256": {role: hashlib.sha256(text.encode()).hexdigest() for role, text in prompts.items()},
            "schemas": {role: schema.model_json_schema() for role, schema in schemas.items()},
        },
        "environment": environment, "models": [], "complete": False,
    }
    with ModelRouter(work_dir=args.output.parent / "benchmark-runtime", port=args.port) as router:
        for key in selected:
            model = config["models"][key]
            entry = {"model_key": key, **model, "cases": []}
            print("Модель:", model["name"], flush=True)
            for case in cases():
                record = {"id": case["id"], "task_type": case["role"], "expected": case["expected"],
                          "payload": case["payload"], "correct": False, "schema_valid": False}
                try:
                    generated = router.generate_model(key, prompts[case["role"]], case["payload"], schemas[case["role"]].model_json_schema())
                    record.update(generated)
                    schemas[case["role"]].model_validate(generated["data"])
                    record["schema_valid"] = True
                    record["correct"] = check_answer(case, generated["data"])
                except (RuntimeError, ValidationError) as exc:
                    record["error"] = str(exc)
                entry["cases"].append(record)
                print(case["id"], record["correct"], record.get("inference_s"), flush=True)
            good = [r for r in entry["cases"] if "inference_s" in r]
            entry["aggregate"] = {
                "n_cases": len(entry["cases"]),
                "task_accuracy": statistics.mean(r["correct"] for r in entry["cases"]),
                "schema_accuracy": statistics.mean(r["schema_valid"] for r in entry["cases"]),
                "mean_inference_s": statistics.mean(r["inference_s"] for r in good) if good else None,
                "p95_inference_s": float(np.quantile([r["inference_s"] for r in good], 0.95)) if good else None,
                "mean_tokens_per_second": statistics.mean(r["tokens_per_second"] for r in good) if good else None,
                "cold_load_s": sum(r.get("load_s", 0) for r in entry["cases"]),
            }
            for role in ("planner", "writer", "validator"):
                group = [r for r in entry["cases"] if r["task_type"] == role]
                entry["aggregate"][role + "_accuracy"] = statistics.mean(r["correct"] for r in group)
            negative = [r for r in entry["cases"] if r["expected"].get("approved") is False]
            entry["aggregate"]["false_approval_rate"] = statistics.mean(
                r.get("data", {}).get("approved") is True for r in negative)
            result["models"].append(entry)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    result["completed_calls"] = sum("raw_content" in case for entry in result["models"] for case in entry["cases"])
    result["expected_calls"] = len(selected) * len(cases())
    result["complete"] = result["completed_calls"] == result["expected_calls"]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise RuntimeError("Benchmark не завершил все настоящие вызовы; см. ошибки в JSON")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        columns = ["model_key", "name", "family", "quantization", "size_bytes", *result["models"][0]["aggregate"]]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in result["models"]:
            writer.writerow({**{name: entry[name] for name in columns[:5]}, **entry["aggregate"]})
    plot_summary(result["models"], args.output.with_suffix(".png"))
    print("Сохранено:", args.output)


if __name__ == "__main__":
    main()
