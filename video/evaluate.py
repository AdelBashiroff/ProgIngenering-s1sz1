"""Оценка качества детекции объектов (mAP) на подмножестве COCO val2017.

Сначала скачайте данные:  python download_coco_val.py
Запуск:
    python evaluate.py                                        # yolo11n.pt на всех скачанных изображениях
    python evaluate.py --models yolo11n.pt yolo11s.pt         # сравнение моделей
    python evaluate.py --n 100                                # быстрая проверка

Метрики (стандарт COCO, bbox): AP@[.5:.95], AP50, AP75, AP по размерам объектов (small/medium/large),
AR, AP по каждому классу, а также скорость (мс на изображение) и размер модели.
Модель обучена на COCO train2017, val2017 для обучения не использовалась.

Результаты сохраняются в results/: metrics_<модель>.json, per_class_ap_<модель>.csv, comparison.csv.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detector import DEFAULT_MODEL, get_device, get_model

BASE = Path(__file__).parent
COCO_DIR = BASE / "data" / "coco"
STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]


def load_gt(coco_dir: Path, n):
    ann = coco_dir / "annotations" / "instances_val2017.json"
    img_dir = coco_dir / "val2017"
    if not ann.exists() or not img_dir.exists():
        raise SystemExit("Данные COCO не найдены. Сначала выполните: python download_coco_val.py")
    coco = COCO(str(ann))
    ids = sorted(i for i, info in coco.imgs.items() if (img_dir / info["file_name"]).exists())
    if not ids:
        raise SystemExit("В data/coco/val2017 нет изображений. Выполните: python download_coco_val.py")
    if n:
        ids = ids[:n]
    return coco, ids, img_dir


def class_mapping(model, coco) -> list:
    """Индекс класса YOLO -> category_id COCO (порядок классов совпадает; проверяем по названиям)."""
    cat_ids = sorted(coco.getCatIds())
    for i, cid in enumerate(cat_ids):
        if model.names[i].lower() != coco.cats[cid]["name"].lower():
            print(f"Внимание: класс {i} модели ({model.names[i]}) не совпадает с COCO ({coco.cats[cid]['name']})")
    return cat_ids


def evaluate_model(model_name: str, coco, ids, img_dir: Path, imgsz: int, out_dir: Path) -> dict:
    model = get_model(model_name)
    device = get_device()
    cat_ids = class_mapping(model, coco)
    n_params = sum(p.numel() for p in model.model.parameters())

    first = img_dir / coco.imgs[ids[0]]["file_name"]
    model.predict(str(first), imgsz=imgsz, device=device, verbose=False)  # прогрев

    detections, times = [], []
    for k, img_id in enumerate(ids, 1):
        path = img_dir / coco.imgs[img_id]["file_name"]
        t0 = time.perf_counter()
        # низкий порог уверенности и max_det=300 — стандартные настройки для расчёта mAP
        result = model.predict(str(path), conf=0.001, iou=0.7, imgsz=imgsz, max_det=300,
                               device=device, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            x1, y1, x2, y2 = xyxy
            detections.append({"image_id": img_id, "category_id": cat_ids[int(cls)],
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
        if k % 100 == 0:
            print(f"  обработано {k}/{len(ids)}")
    if not detections:
        raise SystemExit("Модель не нашла ни одного объекта — оценка невозможна")

    coco_dt = coco.loadRes(detections)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # AP по классам: precision[T, R, K, A, M], A=0 (все размеры), M=2 (maxDets=100)
    precision = ev.eval["precision"]
    rows = []
    for k, cid in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        n_gt = len(coco.getAnnIds(imgIds=ids, catIds=[cid], iscrowd=None))
        rows.append({"class": coco.cats[cid]["name"], "AP": float(p.mean()) if p.size else float("nan"),
                     "n_objects": n_gt})
    per_class = pd.DataFrame(rows).sort_values("AP", ascending=False)
    ranked = per_class.dropna(subset=["AP"])  # классы, которых нет в выборке, не участвуют в рейтинге

    metrics = {
        "model": model_name,
        "device": device,
        "n_images": len(ids),
        "imgsz": imgsz,
        **{name: round(float(v), 4) for name, v in zip(STAT_NAMES, ev.stats)},
        "ms_per_image": round(float(np.mean(times)) * 1000, 1),
        "images_per_second": round(1 / float(np.mean(times)), 2),
        "parameters_millions": round(n_params / 1e6, 2),
    }
    safe = model_name.replace("/", "__").replace(".pt", "")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"metrics_{safe}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class.round(4).to_csv(out_dir / f"per_class_ap_{safe}.csv", index=False)

    print(f"\n=== {model_name} ({device}) ===")
    print(f"Изображений: {metrics['n_images']}")
    print(f"AP@[.5:.95]: {metrics['AP']}   AP50: {metrics['AP50']}   AP75: {metrics['AP75']}")
    print(f"AP по размерам: small {metrics['AP_small']}, medium {metrics['AP_medium']}, large {metrics['AP_large']}")
    print(f"Скорость: {metrics['ms_per_image']} мс/изображение ({metrics['images_per_second']} изобр./с), "
          f"параметров: {metrics['parameters_millions']} млн")
    print("Лучшие классы:  ", ", ".join(f"{r['class']} {r['AP']:.2f}" for _, r in ranked.head(5).iterrows()))
    print("Худшие классы:  ", ", ".join(f"{r['class']} {r['AP']:.2f}" for _, r in ranked.tail(5).iterrows()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=COCO_DIR)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--n", type=int, default=None, help="взять только первые N изображений")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    coco, ids, img_dir = load_gt(args.data_dir, args.n)
    print(f"Изображений для оценки: {len(ids)}")
    out_dir = BASE / "results"
    all_metrics = [evaluate_model(m, coco, ids, img_dir, args.imgsz, out_dir) for m in args.models]

    if len(all_metrics) > 1:
        table = pd.DataFrame(all_metrics)[["model", "AP", "AP50", "AP75", "ms_per_image", "parameters_millions"]]
        table.to_csv(out_dir / "comparison.csv", index=False)
        print("\n=== Сравнение моделей ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
