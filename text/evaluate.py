"""Оценка качества моделей тональности на собственном размеченном наборе.

Запуск:
    python evaluate.py                                  # обе модели, data/test_reviews.csv
    python evaluate.py --models seara/rubert-tiny2-russian-sentiment
    python evaluate.py --data data/my_reviews.csv

Формат CSV: колонки text,label (label: positive / negative / neutral).

Что сохраняется в results/:
    metrics_<модель>.json            метрики, 95% доверительные интервалы, анализ порога уверенности
    predictions_<модель>.csv         все предсказания (text, label, predicted, score, correct)
    errors_<модель>.csv              только ошибочные предсказания
    confusion_matrix_<модель>.png    матрица ошибок
    comparison.json                  парный bootstrap-тест разницы macro-F1 двух первых моделей
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sentiment import BASE_MODEL, DEFAULT_MODEL, LABELS, get_classifier, predict, predict_batch

BASE = Path(__file__).parent
N_BOOTSTRAP = 1000
SEED = 42
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit("В CSV должны быть колонки: text, label")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].str.strip().str.lower()
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise SystemExit(f"Неизвестные метки в данных: {unknown}. Допустимо: {LABELS}")
    return df.reset_index(drop=True)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)


def _ci(values) -> list:
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def bootstrap_ci(y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """95% доверительные интервалы для accuracy и macro-F1 (bootstrap по примерам)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    accs = np.array([(y_true[i] == y_pred[i]).mean() for i in idx])
    f1s = np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])
    return {"accuracy_ci95": _ci(accs), "macro_f1_ci95": _ci(f1s)}


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Парный bootstrap: разница macro-F1 (A - B) на одних и тех же выборках примеров."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    diffs = np.array([macro_f1(y_true[i], pred_a[i]) - macro_f1(y_true[i], pred_b[i]) for i in idx])
    lo, hi = _ci(diffs)
    return {
        "macro_f1_diff_mean": round(float(diffs.mean()), 4),
        "macro_f1_diff_ci95": [lo, hi],
        "significant_at_95": not (lo <= 0.0 <= hi),
    }


def threshold_analysis(y_true, y_pred, scores) -> list:
    """Как меняются охват и точность, если отбрасывать ответы с низкой уверенностью."""
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
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title, fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def evaluate(model_name: str, df: pd.DataFrame, out_dir: Path):
    get_classifier(model_name)  # загрузка модели не попадает в замер времени
    predict("Прогрев модели", model_name)

    start = time.perf_counter()
    preds = predict_batch(df["text"].tolist(), model_name)
    elapsed = time.perf_counter() - start

    y_true = df["label"].tolist()
    y_pred = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()

    metrics = {
        "model": model_name,
        "n_samples": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        **bootstrap_ci(y_true, y_pred),
        "per_class": classification_report(
            y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": {"labels": list(LABELS), "matrix": cm},
        "confidence_threshold_analysis": threshold_analysis(y_true, y_pred, scores),
        "texts_per_second": round(len(df) / elapsed, 2),
        "ms_per_text": round(elapsed / len(df) * 1000, 1),
    }

    safe_name = model_name.replace("/", "__")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"metrics_{safe_name}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    full = df.assign(predicted=y_pred, score=scores)
    full["correct"] = full["label"] == full["predicted"]
    full.to_csv(out_dir / f"predictions_{safe_name}.csv", index=False)
    errors = full[~full["correct"]].drop(columns="correct")
    errors.to_csv(out_dir / f"errors_{safe_name}.csv", index=False)
    save_confusion_png(cm, model_name.split("/")[-1], out_dir / f"confusion_matrix_{safe_name}.png")

    print(f"\n=== {model_name} ===")
    print(f"Примеров: {metrics['n_samples']}")
    print(f"Accuracy: {metrics['accuracy']}  (95% ДИ {metrics['accuracy_ci95']})")
    print(f"Macro-F1: {metrics['macro_f1']}  (95% ДИ {metrics['macro_f1_ci95']})")
    print(f"Скорость: {metrics['texts_per_second']} текстов/с ({metrics['ms_per_text']} мс/текст)")
    print(classification_report(y_true, y_pred, labels=list(LABELS), zero_division=0))
    print("Матрица ошибок (строки — истина, столбцы — предсказание):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print("\nОтбрасывание неуверенных ответов (score < порога):")
    print(pd.DataFrame(metrics["confidence_threshold_analysis"]).to_string(index=False))
    print(f"\nОшибок: {len(errors)} (results/errors_{safe_name}.csv)")
    return metrics, y_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BASE / "data" / "test_reviews.csv")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, BASE_MODEL])
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Загружено {len(df)} отзывов. Распределение классов:\n{df['label'].value_counts()}")

    out_dir = BASE / "results"
    all_preds = {}
    for model_name in args.models:
        _, y_pred = evaluate(model_name, df, out_dir)
        all_preds[model_name] = y_pred

    if len(args.models) >= 2:
        a, b = args.models[0], args.models[1]
        diff = {"model_a": a, "model_b": b, **paired_bootstrap_diff(df["label"].tolist(), all_preds[a], all_preds[b])}
        (out_dir / "comparison.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== Сравнение: {a.split('/')[-1]} − {b.split('/')[-1]} (macro-F1) ===")
        print(f"Средняя разница: {diff['macro_f1_diff_mean']}, 95% ДИ {diff['macro_f1_diff_ci95']}")
        print("Разница статистически значима" if diff["significant_at_95"]
              else "Разница в пределах погрешности (ДИ содержит 0)")


if __name__ == "__main__":
    main()
