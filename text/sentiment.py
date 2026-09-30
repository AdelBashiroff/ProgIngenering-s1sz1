"""Анализ тональности русскоязычных отзывов (Hugging Face Transformers).

Вход:  текст отзыва (str)
Выход: {"label": "positive" | "negative" | "neutral", "score": float}

Использование из командной строки:
    python sentiment.py "Отличный сервис, всем рекомендую!"

Модель по умолчанию — rubert-tiny2 (выбрана по результатам оценки на собственном
наборе: качество не хуже base-версии, критерий из ТЗ выполняется, модель компактнее).
"""
import sys
from functools import lru_cache

import torch
from transformers import pipeline

DEFAULT_MODEL = "seara/rubert-tiny2-russian-sentiment"
BASE_MODEL = "seara/rubert-base-cased-russian-sentiment"
LABELS = ("positive", "negative", "neutral")
MAX_LENGTH = 256


def _device() -> str:
    """Выбирает лучшее доступное устройство: CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def get_classifier(model_name: str = DEFAULT_MODEL):
    """Загружает модель один раз и кэширует её."""
    return pipeline("text-classification", model=model_name, device=_device())


def predict(text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Определяет тональность одного отзыва."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Текст отзыва не должен быть пустым")
    result = get_classifier(model_name)(text, truncation=True, max_length=MAX_LENGTH)[0]
    return {"label": result["label"], "score": round(float(result["score"]), 4)}


def predict_batch(texts, model_name: str = DEFAULT_MODEL, batch_size: int = 16) -> list:
    """Определяет тональность списка отзывов (быстрее, чем по одному)."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    results = get_classifier(model_name)(
        texts, batch_size=batch_size, truncation=True, max_length=MAX_LENGTH
    )
    return [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in results]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python sentiment.py "текст отзыва"')
        sys.exit(1)
    print(predict(" ".join(sys.argv[1:])))
