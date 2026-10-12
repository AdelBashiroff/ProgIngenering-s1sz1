"""Zero-shot классификация скриншотов сайтов с помощью CLIP (Hugging Face Transformers, PyTorch).

CLIP умеет сопоставлять картинки и тексты. Мы описываем каждый класс текстом (промптом) и выбираем тот
класс, текст которого лучше всего подходит к скриншоту. Обучение не требуется.

Вход:  изображение (путь к файлу или PIL.Image)
Выход: {"label": "landing" | "shop" | "blog", "score": float, "probabilities": {...}}

Использование из командной строки:
    python clip_classifier.py data/images/blog/001_example-com.jpg
"""
import sys
from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from .labels import LABELS
except ImportError:  # запуск python image/clip_classifier.py
    from labels import LABELS

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
ALT_MODEL = "openai/clip-vit-base-patch16"
DEFAULT_FIT = "pad"          # как приводить скриншот к квадрату: pad (поля) или crop (обрезка по центру)
DEFAULT_PROMPTS = "descriptive"

# Наборы текстовых описаний классов (порядок классов совпадает с LABELS).
PROMPT_SETS = {
    # короткие описания
    "simple": {
        "landing": ["a screenshot of a landing page"],
        "shop": ["a screenshot of an online store"],
        "blog": ["a screenshot of a blog"],
    },
    # подробные описания с характерными элементами
    "descriptive": {
        "landing": ["a screenshot of a landing page with a headline, a call-to-action button and marketing sections"],
        "shop": ["a screenshot of an online store with product cards, prices and a shopping cart"],
        "blog": ["a screenshot of a blog page with article titles, text and an author"],
    },
    # несколько формулировок на класс; эмбеддинги усредняются (ансамбль промптов)
    "ensemble": {
        "landing": ["a screenshot of a landing page",
                    "a screenshot of a marketing website with a big headline and a sign up button",
                    "a screenshot of a product promo page with a call to action"],
        "shop": ["a screenshot of an online store",
                 "a screenshot of an e-commerce website with products and prices",
                 "a screenshot of a web shop with a product catalog and a cart button"],
        "blog": ["a screenshot of a blog",
                 "a screenshot of a website with articles and blog posts",
                 "a screenshot of a news or article page with long text"],
    },
}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def get_model(model_name: str = DEFAULT_MODEL):
    """Загружает модель и препроцессор один раз."""
    device = _device()
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def logit_scale(model_name: str = DEFAULT_MODEL) -> float:
    """Температура CLIP (обученный множитель сходства, около 100)."""
    model, _, _ = get_model(model_name)
    return float(model.logit_scale.exp().item())


def prepare_image(image: Image.Image, fit: str = DEFAULT_FIT) -> Image.Image:
    """Скриншот широкий, а CLIP принимает квадрат.
    crop — штатная обработка CLIP: обрезка по центру (теряются края страницы);
    pad  — добавляются белые поля сверху и снизу, видна вся страница, но в уменьшенном виде."""
    image = image.convert("RGB")
    if fit == "pad":
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return canvas
    if fit != "crop":
        raise ValueError("fit должен быть 'pad' или 'crop'")
    return image


@torch.no_grad()
def encode_images(images, model_name: str = DEFAULT_MODEL, fit: str = DEFAULT_FIT, batch_size: int = 16):
    """Нормированные эмбеддинги изображений, тензор (N, D) на CPU."""
    model, processor, device = get_model(model_name)
    out = []
    for i in range(0, len(images), batch_size):
        batch = [prepare_image(im, fit) for im in images[i:i + batch_size]]
        pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        feats = model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
@lru_cache(maxsize=12)
def encode_prompts(prompt_set: str = DEFAULT_PROMPTS, model_name: str = DEFAULT_MODEL):
    """Нормированные эмбеддинги классов, тензор (число классов, D). Несколько промптов усредняются."""
    model, processor, device = get_model(model_name)
    rows = []
    for label in LABELS:
        tokens = processor(text=PROMPT_SETS[prompt_set][label], padding=True, truncation=True,
                           return_tensors="pt").to(device)
        feats = model.text_projection(model.text_model(**tokens).pooler_output)
        feats = torch.nn.functional.normalize(feats, dim=-1).mean(dim=0)
        rows.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
    return torch.stack(rows)


def classify_features(image_feats, text_feats, scale: float):
    """Вероятности классов (softmax по сходству), тензор (N, число классов)."""
    return torch.softmax(scale * image_feats @ text_feats.T, dim=-1)


def predict(image, model_name: str = DEFAULT_MODEL, prompts: str = DEFAULT_PROMPTS, fit: str = DEFAULT_FIT) -> dict:
    """Определяет тип сайта по скриншоту (путь к файлу или PIL.Image)."""
    if prompts not in PROMPT_SETS:
        raise ValueError(f"Неизвестный набор промптов: {prompts}")
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    probs = classify_features(encode_images([image], model_name, fit),
                              encode_prompts(prompts, model_name), logit_scale(model_name))[0]
    best = int(probs.argmax())
    return {
        "label": LABELS[best],
        "score": round(float(probs[best]), 4),
        "probabilities": {label: round(float(p), 4) for label, p in zip(LABELS, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python clip_classifier.py путь/к/скриншоту.jpg")
        sys.exit(1)
    print(predict(sys.argv[1]))
