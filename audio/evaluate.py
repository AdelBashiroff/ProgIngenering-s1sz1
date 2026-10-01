"""Оценка качества классификации звуков (YAMNet) на подмножестве ESC-10.

Сначала скачайте данные:  python download_esc10.py
Запуск:
    python evaluate.py                       # 400 записей, агрегация по кадрам: среднее
    python evaluate.py --aggregate max       # агрегация по кадрам: максимум
    python evaluate.py --per-class 5         # быстрая проверка на 50 записях

Модель не обучается, поэтому для оценки используются все записи (фолды ESC-50 не нужны).

Метрики:
  - закрытая задача (10 классов): accuracy, macro-F1, метрики по классам, матрица ошибок,
    95% доверительные интервалы (bootstrap);
  - анализ порога оценки: охват и точность при отбрасывании записей с низким score;
  - открытая задача (все 521 класс YAMNet): top-1 и top-5 — попадает ли лучший (или один из
    пяти лучших) класс YAMNet в набор классов, сопоставленных истинному классу ESC-10;
  - скорость: миллисекунд на запись (чтение файла + инференс).

Результаты сохраняются в results/ (с суффиксом способа агрегации).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import yamnet_classifier as yc

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)  # матрица ошибок печатается целиком

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "ESC-50"
LABELS = list(yc.ESC10_CLASSES)
N_BOOTSTRAP = 1000
SEED = 42
THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def load_esc10(data_dir: Path, per_class=None) -> pd.DataFrame:
    meta_path = data_dir / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise SystemExit("Данные не найдены. Сначала выполните: python download_esc10.py")
    meta = pd.read_csv(meta_path)
    meta = meta[meta["esc10"]].copy()
    meta["label"] = meta["category"]
    unknown = set(meta["label"]) - set(LABELS)
    if unknown:
        raise SystemExit(f"Классы из данных отсутствуют в сопоставлении: {unknown}")
    if per_class:
        meta = meta.groupby("label").head(per_class)
    meta["path"] = meta["filename"].map(lambda f: data_dir / "audio" / f)
    missing = [str(p) for p in meta["path"] if not p.exists()]
    if missing:
        raise SystemExit(f"Не найдено файлов: {len(missing)} (например, {missing[0]}). "
                         "Выполните: python download_esc10.py")
    return meta.reset_index(drop=True)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """95% доверительные интервалы для accuracy и macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    accs = np.array([(y_true[i] == y_pred[i]).mean() for i in idx])
    f1s = np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    return {"accuracy_ci95": ci(accs), "macro_f1_ci95": ci(f1s)}


def threshold_analysis(y_true, y_pred, scores) -> list:
    """Как меняются охват и точность, если отбрасывать записи с низкой оценкой (score)."""
    y_true, y_pred, scores = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores)
    rows = []
    for t in THRESHOLDS:
        mask = scores >= t
        n = int(mask.sum())
        rows.append({
            "threshold": t,
            "coverage": round(n / len(y_true), 4),
            "accuracy": round(float((y_true[mask] == y_pred[mask]).mean()), 4) if n else None,
        })
    return rows


def save_confusion_png(matrix, title: str, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — картинка матрицы ошибок пропущена")
        return False
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title, fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--aggregate", choices=["mean", "max"], default="mean",
                        help="как объединять оценки по кадрам (по умолчанию mean)")
    parser.add_argument("--per-class", type=int, default=None, help="взять только N записей на класс")
    args = parser.parse_args()

    df = load_esc10(args.data_dir, args.per_class)
    print(f"Записей: {len(df)}. Агрегация по кадрам: {args.aggregate}")
    print(df["label"].value_counts().to_string())

    class_names = yc.get_class_names()
    mapped = yc.mapped_indices(class_names)
    mapped_sets = {c: set(ids) for c, ids in mapped.items()}

    yc.scores_for_file(df["path"].iloc[0], args.aggregate)  # прогрев, не входит в замер времени

    rows = []
    start = time.perf_counter()
    for n, (path, label) in enumerate(zip(df["path"], df["label"]), 1):
        scores = yc.scores_for_file(path, args.aggregate)
        pred, score = yc.esc10_prediction(scores, class_names)
        top5 = np.argsort(scores)[::-1][:5]
        rows.append({
            "filename": path.name,
            "label": label,
            "predicted": pred,
            "score": round(score, 4),
            "yamnet_top1": class_names[top5[0]],
            "yamnet_top1_score": round(float(scores[top5[0]]), 4),
            "top1_in_class": int(top5[0]) in mapped_sets[label],
            "top5_in_class": any(int(i) in mapped_sets[label] for i in top5),
        })
        if n % 50 == 0:
            print(f"  обработано {n}/{len(df)}")
    elapsed = time.perf_counter() - start

    res = pd.DataFrame(rows)
    res["correct"] = res["label"] == res["predicted"]
    y_true, y_pred = res["label"].tolist(), res["predicted"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    metrics = {
        "aggregate": args.aggregate,
        "n_samples": len(res),
        "closed_set_10_classes": {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(macro_f1(y_true, y_pred), 4),
            **bootstrap_ci(y_true, y_pred),
            "per_class": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": {"labels": LABELS, "matrix": cm},
            "confidence_threshold_analysis": threshold_analysis(y_true, y_pred, res["score"].tolist()),
        },
        "open_set_521_classes": {
            "top1_in_mapped_classes": round(float(res["top1_in_class"].mean()), 4),
            "top5_in_mapped_classes": round(float(res["top5_in_class"].mean()), 4),
        },
        "ms_per_clip": round(elapsed / len(res) * 1000, 1),
        "clips_per_second": round(len(res) / elapsed, 2),
    }

    out = BASE / "results"
    out.mkdir(exist_ok=True)
    suffix = args.aggregate
    (out / f"metrics_{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out / f"predictions_{suffix}.csv", index=False)
    res[~res["correct"]].drop(columns="correct").to_csv(out / f"errors_{suffix}.csv", index=False)
    save_confusion_png(cm, f"YAMNet, ESC-10 (агрегация: {args.aggregate})", out / f"confusion_matrix_{suffix}.png")

    c = metrics["closed_set_10_classes"]
    print("\n=== Закрытая задача: 10 классов ===")
    print(f"Accuracy: {c['accuracy']}  (95% ДИ {c['accuracy_ci95']})")
    print(f"Macro-F1: {c['macro_f1']}  (95% ДИ {c['macro_f1_ci95']})")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Матрица ошибок (строки — истина, столбцы — предсказание):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print("\nОтбрасывание записей с низкой оценкой (score < порога):")
    print(pd.DataFrame(c["confidence_threshold_analysis"]).to_string(index=False))
    o = metrics["open_set_521_classes"]
    print("\n=== Открытая задача: все 521 класс YAMNet ===")
    print(f"Top-1 в сопоставленных классах: {o['top1_in_mapped_classes']}")
    print(f"Top-5 в сопоставленных классах: {o['top5_in_mapped_classes']}")
    print(f"\nСкорость: {metrics['ms_per_clip']} мс/запись ({metrics['clips_per_second']} записей/с)")
    print(f"Ошибок: {int((~res['correct']).sum())} (results/errors_{suffix}.csv)")


if __name__ == "__main__":
    main()
