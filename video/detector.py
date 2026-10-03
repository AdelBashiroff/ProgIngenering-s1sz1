"""Детекция объектов на видео с помощью предобученной модели YOLO (Ultralytics, PyTorch).

Вход:  видеофайл (mp4, avi, mov ...)
Выход: видео с рамками и подписями + таблица всех детекций (CSV) + сводка (JSON)

Использование из командной строки:
    python detector.py videos/clip.mp4
    python detector.py videos/clip.mp4 --model yolo11s.pt --conf 0.4 --max-frames 300

Модель использует 80 классов набора COCO (person, car, dog, ...). Обучение не требуется.
"""
import argparse
import csv
import json
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import torch

BASE = Path(__file__).parent
DEFAULT_MODEL = "yolo11n.pt"


def get_device() -> str:
    """Выбирает лучшее доступное устройство: CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def get_model(model_name: str = DEFAULT_MODEL):
    """Загружает модель один раз. Веса скачиваются автоматически в папку weights/."""
    from ultralytics import YOLO

    weights = BASE / "weights" / model_name
    weights.parent.mkdir(exist_ok=True)
    return YOLO(str(weights))


def _to_detections(result, names) -> list:
    detections = []
    for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        detections.append({
            "label": names[int(cls)],
            "confidence": round(float(conf), 4),
            "box": [round(v, 1) for v in box],  # x1, y1, x2, y2 в пикселях
        })
    return detections


def detect_frame(frame, model_name: str = DEFAULT_MODEL, conf: float = 0.25, imgsz: int = 640) -> list:
    """Детекция объектов на одном кадре (массив BGR, как в OpenCV)."""
    model = get_model(model_name)
    result = model.predict(frame, conf=conf, imgsz=imgsz, device=get_device(), verbose=False)[0]
    return _to_detections(result, model.names)


def process_video(input_path, output_path=None, model_name: str = DEFAULT_MODEL, conf: float = 0.25,
                  imgsz: int = 640, max_frames=None, fourcc: str = "mp4v") -> dict:
    """Обрабатывает видео: рисует рамки, сохраняет видео, CSV с детекциями и JSON со сводкой."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)
    stem = input_path.stem
    output_path = Path(output_path) if output_path else results_dir / f"{stem}_detected.mp4"
    csv_path = results_dir / f"{stem}_detections.csv"
    json_path = results_dir / f"{stem}_summary.json"

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Не удалось создать выходное видео (попробуйте другой --fourcc, например avc1)")

    model = get_model(model_name)
    device = get_device()

    ok, first = cap.read()  # прогрев модели: не входит в замер скорости
    if ok:
        model.predict(first, conf=conf, imgsz=imgsz, device=device, verbose=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_det = Counter()
    frames_with = Counter()
    max_simul = Counter()
    infer_times = []
    n = 0
    start = time.perf_counter()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f, lineterminator="\n")
        csv_writer.writerow(["frame", "time_s", "label", "confidence", "x1", "y1", "x2", "y2"])
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n >= max_frames):
                break
            t0 = time.perf_counter()
            result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            infer_times.append(time.perf_counter() - t0)

            detections = _to_detections(result, model.names)
            counts = Counter(d["label"] for d in detections)
            for label, c in counts.items():
                total_det[label] += c
                frames_with[label] += 1
                max_simul[label] = max(max_simul[label], c)
            for d in detections:
                csv_writer.writerow([n, round(n / fps, 3), d["label"], d["confidence"], *d["box"]])

            writer.write(result.plot())  # кадр с нарисованными рамками
            n += 1
            if n % 50 == 0:
                print(f"  обработано кадров: {n}" + (f" из {total}" if total > 0 else ""))

    elapsed = time.perf_counter() - start
    cap.release()
    writer.release()
    if n == 0:
        raise ValueError("В видео не найдено ни одного кадра")

    processing_fps = n / elapsed
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "model": model_name,
        "device": device,
        "conf_threshold": conf,
        "imgsz": imgsz,
        "frames": n,
        "resolution": [width, height],
        "video_fps": round(fps, 2),
        "duration_s": round(n / fps, 2),
        "processing_fps": round(processing_fps, 2),
        "realtime_factor": round(processing_fps / fps, 2),  # >1 — быстрее реального времени
        "inference_ms_per_frame": round(sum(infer_times) / n * 1000, 1),
        "detections_total": int(sum(total_det.values())),
        "classes": {
            label: {
                "detections": total_det[label],
                "frames_with_class": frames_with[label],
                "max_simultaneous": max_simul[label],
            }
            for label, _ in total_det.most_common()
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="путь к видеофайлу")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить видео с рамками")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="веса YOLO (yolo11n.pt, yolo11s.pt, ...)")
    parser.add_argument("--conf", type=float, default=0.25, help="порог уверенности (0–1)")
    parser.add_argument("--imgsz", type=int, default=640, help="размер входа модели")
    parser.add_argument("--max-frames", type=int, default=None, help="обработать только первые N кадров")
    parser.add_argument("--fourcc", default="mp4v", help="кодек выходного видео")
    args = parser.parse_args()

    summary = process_video(args.video, args.out, args.model, args.conf, args.imgsz, args.max_frames, args.fourcc)
    print(f"\nГотово: {summary['output']}")
    print(f"Кадров: {summary['frames']} ({summary['duration_s']} с, {summary['resolution'][0]}x{summary['resolution'][1]}), "
          f"устройство: {summary['device']}")
    print(f"Скорость: {summary['processing_fps']} кадров/с (x{summary['realtime_factor']} от реального времени), "
          f"инференс {summary['inference_ms_per_frame']} мс/кадр")
    print(f"Всего детекций: {summary['detections_total']}")
    for label, st in list(summary["classes"].items())[:10]:
        print(f"  {label:15} детекций: {st['detections']:5}  кадров с классом: {st['frames_with_class']:5}  "
              f"макс. одновременно: {st['max_simultaneous']}")


if __name__ == "__main__":
    main()
