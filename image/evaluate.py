"""Оценка zero-shot классификации скриншотов сайтов (CLIP) на собственном размеченном наборе.

Сначала соберите данные:  python capture_screenshots.py
Запуск:
    python evaluate.py                                  # модель ViT-B/32, все варианты промптов и обработки
    python evaluate.py --models openai/clip-vit-base-patch32 openai/clip-vit-base-patch16
    python evaluate.py --prompts simple descriptive --fit pad

Перебираются конфигурации: модель x способ приведения к квадрату (pad/crop) x набор текстовых описаний
(simple/descriptive/ensemble). Эмбеддинги изображений считаются один раз на пару (модель, способ).

Сохраняется в results/:
    comparison.csv                  все конфигурации: accuracy, macro-F1
    metrics_<конфигурация>.json     подробные метрики (для конфигурации по умолчанию и лучшей по macro-F1)
    predictions_<конфигурация>.csv  предсказания для каждого скриншота
    errors_<конфигурация>.csv       только ошибки
    confusion_matrix_<конфигурация>.png

ВАЖНО: лучшая конфигурация выбирается по тем же данным, на которых считается качество, поэтому её результат
немного оптимистичен. Для честной картины в отчёте приводится таблица всех конфигураций.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from . import clip_classifier as cc
    from .labels import LABELS
except ImportError:  # запуск python image/evaluate.py
    import clip_classifier as cc
    from labels import LABELS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
LABEL_LIST = list(LABELS)
N_BOOTSTRAP = 1000
SEED = 42
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_PER_CLASS = 20


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Читает data/manifest.csv; если его нет, собирает набор из папок data/images/<класс>/."""
    manifest = data_dir / "manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        df = df[df["status"].isin(["ok", "cached"])].copy()
        if "qa" in df.columns:
            df = df[df["qa"] == "accept"].copy()
        df["path"] = df["file"].map(lambda f: data_dir / "images" / f)
    else:
        rows = [{"label": folder.name, "file": f"{folder.name}/{p.name}", "path": p}
                for folder in sorted((data_dir / "images").glob("*")) if folder.is_dir() and folder.name in LABELS
                for p in sorted(folder.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Скриншоты не найдены. Сначала выполните: python capture_screenshots.py")
    exists = df["path"].map(lambda p: Path(p).exists())
    if (~exists).any():
        print(f"Пропущено записей без файла (удалены вручную?): {int((~exists).sum())}")
    df = df[exists].reset_index(drop=True)
    if df.empty:
        raise SystemExit("В манифесте нет доступных изображений")
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise SystemExit(f"Неизвестные классы в данных: {unknown}")
    return df


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=LABEL_LIST, average="macro", zero_division=0)


def _ci(values) -> list:
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def bootstrap_ci(y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """95% доверительные интервалы для accuracy и macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    accs = np.array([(y_true[i] == y_pred[i]).mean() for i in idx])
    f1s = np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])
    return {"accuracy_ci95": _ci(accs), "macro_f1_ci95": _ci(f1s)}


def threshold_analysis(y_true, y_pred, scores) -> list:
    """Охват и точность при отбрасывании ответов с низкой уверенностью."""
    y_true, y_pred, scores = np.asarray(y_true), np.asarray(y_pred), np.asarray(scores)
    rows = []
    for t in THRESHOLDS:
        mask = scores >= t
        n = int(mask.sum())
        rows.append({"threshold": t, "coverage": round(n / len(y_true), 4),
                     "accuracy": round(float((y_true[mask] == y_pred[mask]).mean()), 4) if n else None})
    return rows


def breakdown(df: pd.DataFrame, column: str):
    """Точность по группам (источник ссылки, язык), если такая колонка есть в данных."""
    if column not in df.columns or df[column].isna().all():
        return None
    groups = df.dropna(subset=[column]).groupby(column).agg(n=("correct", "size"), accuracy=("correct", "mean"))
    return groups.round(4).reset_index().to_dict(orient="records")


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
    ax.set_title(title, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def config_name(model_name: str, fit: str, prompts: str) -> str:
    return f"{model_name.split('/')[-1]}__{fit}__{prompts}"


def save_details(name: str, df: pd.DataFrame, y_pred, scores, out_dir: Path):
    y_true = df["label"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_LIST).tolist()
    res = df.drop(columns=["path"], errors="ignore").assign(predicted=y_pred, score=np.round(scores, 4))
    res["correct"] = res["label"] == res["predicted"]

    metrics = {
        "config": name,
        "n_samples": len(df),
        "class_counts": df["label"].value_counts().to_dict(),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        **bootstrap_ci(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=LABEL_LIST, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": LABEL_LIST, "matrix": cm},
        "confidence_threshold_analysis": threshold_analysis(y_true, y_pred, scores),
        "by_source": breakdown(res, "source"),
        "by_lang": breakdown(res, "lang"),
    }
    (out_dir / f"metrics_{name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out_dir / f"predictions_{name}.csv", index=False)
    res[~res["correct"]].drop(columns="correct").to_csv(out_dir / f"errors_{name}.csv", index=False)
    save_confusion_png(cm, name, out_dir / f"confusion_matrix_{name}.png")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']} (95% ДИ {metrics['accuracy_ci95']})   "
          f"Macro-F1: {metrics['macro_f1']} (95% ДИ {metrics['macro_f1_ci95']})")
    print(classification_report(y_true, y_pred, labels=LABEL_LIST, zero_division=0))
    print("Матрица ошибок (строки — истина, столбцы — предсказание):")
    print(pd.DataFrame(cm, index=LABEL_LIST, columns=LABEL_LIST))
    print("\nОтбрасывание ответов с низкой уверенностью:")
    print(pd.DataFrame(metrics["confidence_threshold_analysis"]).to_string(index=False))
    for key, title in (("by_source", "по источникам ссылок"), ("by_lang", "по языкам")):
        if metrics[key]:
            print(f"\nТочность {title}:")
            print(pd.DataFrame(metrics[key]).to_string(index=False))
    print(f"\nОшибок: {int((~res['correct']).sum())} (results/errors_{name}.csv)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--models", nargs="+", default=[cc.DEFAULT_MODEL])
    parser.add_argument("--fit", nargs="+", default=["crop", "pad"], choices=["crop", "pad"])
    parser.add_argument("--prompts", nargs="+", default=list(cc.PROMPT_SETS), choices=list(cc.PROMPT_SETS))
    args = parser.parse_args()

    df = load_dataset(args.data_dir)
    counts = df["label"].value_counts()
    print(f"Скриншотов: {len(df)}. По классам: {counts.to_dict()}")
    for label in LABELS:
        if counts.get(label, 0) < MIN_PER_CLASS:
            print(f"Внимание: в классе {label} меньше {MIN_PER_CLASS} скриншотов — оценка будет неточной")

    images = [Image.open(p).convert("RGB") for p in df["path"]]
    y_true = df["label"].tolist()
    out_dir = BASE / "results"
    out_dir.mkdir(exist_ok=True)

    rows, preds = [], {}
    for model_name in args.models:
        scale = cc.logit_scale(model_name)
        for fit in args.fit:
            start = time.perf_counter()
            image_feats = cc.encode_images(images, model_name, fit)
            ms_per_image = (time.perf_counter() - start) / len(images) * 1000
            for prompt_set in args.prompts:
                probs = cc.classify_features(image_feats, cc.encode_prompts(prompt_set, model_name), scale).numpy()
                y_pred = [LABEL_LIST[i] for i in probs.argmax(axis=1)]
                name = config_name(model_name, fit, prompt_set)
                preds[name] = (y_pred, probs.max(axis=1))
                rows.append({"config": name, "model": model_name.split("/")[-1], "fit": fit, "prompts": prompt_set,
                             "accuracy": round(accuracy_score(y_true, y_pred), 4),
                             "macro_f1": round(macro_f1(y_true, y_pred), 4),
                             "ms_per_image": round(ms_per_image, 1)})

    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print("\n=== Все конфигурации (по убыванию macro-F1) ===")
    print(table.drop(columns="config").to_string(index=False))

    default_name = config_name(cc.DEFAULT_MODEL, cc.DEFAULT_FIT, cc.DEFAULT_PROMPTS)
    detail = list(preds)
    for name in detail:
        label = "конфигурация по умолчанию" if name == default_name else "сравниваемая конфигурация"
        print(f"\n##### Подробно: {label} #####")
        y_pred, scores = preds[name]
        save_details(name, df, y_pred, scores, out_dir)


if __name__ == "__main__":
    main()
