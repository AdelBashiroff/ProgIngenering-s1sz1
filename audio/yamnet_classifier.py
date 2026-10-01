"""Классификация звуков с помощью предобученной модели YAMNet (TensorFlow Hub).

YAMNet предсказывает 521 класс звуковых событий (онтология AudioSet).
Здесь поверх неё сделана задача из ТЗ: распознавание 10 классов окружающих звуков
(подмножество ESC-10) через сопоставление классов ESC-10 -> классы YAMNet. Дообучение не нужно.

Вход:  путь к аудиофайлу WAV (любая частота дискретизации, моно или стерео)
Выход: {"label": <один из 10 классов>, "score": float}

Использование из командной строки:
    python yamnet_classifier.py путь/к/файлу.wav
"""
import sys
from functools import lru_cache
from math import gcd

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000  # YAMNet работает с 16 кГц, моно, float32 в диапазоне [-1, 1]

# Классы ESC-10 -> названия классов YAMNet (display_name из yamnet_class_map.csv)
ESC10_TO_YAMNET = {
    "dog": ["Dog", "Bark"],
    "rooster": ["Chicken, rooster", "Crowing, cock-a-doodle-doo"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "sea_waves": ["Ocean", "Waves, surf"],
    "crackling_fire": ["Fire", "Crackle"],
    "crying_baby": ["Baby cry, infant cry", "Crying, sobbing"],
    "sneezing": ["Sneeze"],
    "clock_tick": ["Tick", "Tick-tock", "Clock"],
    "helicopter": ["Helicopter"],
    "chainsaw": ["Chainsaw"],
}
ESC10_CLASSES = tuple(ESC10_TO_YAMNET)


def load_audio(path) -> np.ndarray:
    """Читает WAV, приводит к моно, float32 [-1, 1] и 16 кГц."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        wav = data / 32768.0
    elif data.dtype == np.int32:
        wav = data / 2147483648.0
    elif data.dtype == np.uint8:
        wav = (data.astype(np.float32) - 128.0) / 128.0
    else:
        wav = data.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(int(sr), SAMPLE_RATE)
        wav = resample_poly(wav, SAMPLE_RATE // g, int(sr) // g)
    return np.asarray(wav, dtype=np.float32)


@lru_cache(maxsize=1)
def _load_model():
    import tensorflow_hub as hub  # импорт здесь, чтобы модуль импортировался без TensorFlow

    model = hub.load(YAMNET_HANDLE)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = tuple(pd.read_csv(class_map_path)["display_name"])
    return model, class_names


def get_class_names() -> tuple:
    """Названия 521 класса YAMNet."""
    return _load_model()[1]


def scores_for_file(path, aggregate: str = "mean") -> np.ndarray:
    """Оценки 521 класса для файла. YAMNet выдаёт оценки по кадрам (~0.48 с), их агрегируют."""
    import tensorflow as tf

    model, _ = _load_model()
    waveform = tf.constant(load_audio(path))
    frame_scores, _, _ = model(waveform)
    frame_scores = frame_scores.numpy()
    if aggregate == "max":
        return frame_scores.max(axis=0)
    return frame_scores.mean(axis=0)


def mapped_indices(class_names) -> dict:
    """Индексы классов YAMNet, соответствующих каждому классу ESC-10."""
    index = {name: i for i, name in enumerate(class_names)}
    missing = [n for names in ESC10_TO_YAMNET.values() for n in names if n not in index]
    if missing:
        raise ValueError(f"Классы не найдены в карте YAMNet: {missing}")
    return {c: [index[n] for n in names] for c, names in ESC10_TO_YAMNET.items()}


def esc10_prediction(scores: np.ndarray, class_names) -> tuple:
    """Лучший из 10 классов: для каждого берётся максимум по сопоставленным классам YAMNet."""
    idx = mapped_indices(class_names)
    class_scores = {c: float(max(scores[i] for i in ids)) for c, ids in idx.items()}
    label = max(class_scores, key=class_scores.get)
    return label, class_scores[label]


def top_k(scores: np.ndarray, class_names, k: int = 5) -> list:
    """k лучших классов YAMNet из всех 521."""
    order = np.argsort(scores)[::-1][:k]
    return [{"label": class_names[i], "score": round(float(scores[i]), 4)} for i in order]


def predict(path, aggregate: str = "mean") -> dict:
    """Определяет класс звука (один из 10). score — оценка YAMNet, не вероятность."""
    class_names = get_class_names()
    label, score = esc10_prediction(scores_for_file(path, aggregate), class_names)
    return {"label": label, "score": round(score, 4)}


def classify(path, k: int = 5, aggregate: str = "mean") -> dict:
    """Подробный результат: класс из 10 и k лучших классов YAMNet из 521."""
    class_names = get_class_names()
    scores = scores_for_file(path, aggregate)
    label, score = esc10_prediction(scores, class_names)
    return {"label": label, "score": round(score, 4), "yamnet_top": top_k(scores, class_names, k)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python yamnet_classifier.py путь/к/файлу.wav")
        sys.exit(1)
    print(classify(sys.argv[1]))
