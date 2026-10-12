"""Запуск из корня: uvicorn api.main:app --host 127.0.0.1 --port 8000."""
import io
import threading
import warnings
from typing import Callable, Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

app = FastAPI(title='Website Screenshot Classifier', version='1.0.0',
              description='Zero-shot CLIP: landing, shop, blog. Score не является калиброванной вероятностью.')
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
_inference_lock = threading.Lock()


class Prediction(BaseModel):
    label: Literal['landing', 'shop', 'blog']
    score: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model: str = 'openai/clip-vit-base-patch32'


def get_predictor() -> Callable:
    # Ленивая загрузка: health и OpenAPI доступны без весов модели.
    from image.clip_classifier import predict
    return predict


@app.get('/health')
def health():
    """Liveness процесса, не проверка наличия весов модели."""
    return {'status': 'ok', 'service': 'website-classifier'}


@app.get('/model')
def model_info():
    return {'model': 'openai/clip-vit-base-patch32', 'labels': ['landing', 'shop', 'blog'],
            'fit': 'pad', 'prompts': 'descriptive', 'max_upload_bytes': MAX_UPLOAD_BYTES,
            'max_image_pixels': MAX_IMAGE_PIXELS}


@app.post('/predict', response_model=Prediction)
def predict_image(file: UploadFile = File(...), predictor: Callable = Depends(get_predictor)):
    """Принимает PNG/JPEG/WebP как multipart/form-data с полем file."""
    try:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'Изображение превышает лимит 8 МиБ')
    if not data:
        raise HTTPException(400, 'Пустой файл')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise HTTPException(415, 'Поддерживаются PNG, JPEG и WebP')
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, 'Изображение превышает лимит 16 млн пикселей')
                source.load()
                image = source.convert('RGB')
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, 'Слишком большое разрешение изображения')
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, 'Файл не является корректным изображением')
    try:
        # Сериализация инференса предотвращает несколько одновременных загрузок весов.
        with _inference_lock:
            result = predictor(image)
    except (OSError, RuntimeError):
        raise HTTPException(503, 'Модель недоступна. Проверьте веса и ресурсы сервера.')
    return Prediction(**result)
