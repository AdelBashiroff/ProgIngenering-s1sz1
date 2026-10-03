"""Скачивает аннотации COCO val2017 и случайную выборку изображений для оценки mAP.

Запуск:
    python download_coco_val.py              # 500 изображений (~80 МБ) + аннотации (скачивается архив ~240 МБ)
    python download_coco_val.py --n 200

Данные сохраняются в data/coco/ (эта папка не должна попадать в Git — см. README).
Выборка случайная, но воспроизводимая (seed=42).
"""
import argparse
import json
import random
import shutil
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "http://images.cocodataset.org"
COCO_DIR = Path(__file__).parent / "data" / "coco"
ANN_ZIP_URL = f"{BASE_URL}/annotations/annotations_trainval2017.zip"
ANN_MEMBER = "annotations/instances_val2017.json"
SEED = 42

CERT_HINT = (
    "Ошибка SSL-сертификата. На macOS запустите «/Applications/Python 3.x/Install Certificates.command» "
    "или выполните: pip install certifi"
)


def _ssl_contexts():
    """Сначала системные сертификаты, затем certifi (запасной вариант для macOS)."""
    yield ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return
    yield ssl.create_default_context(cafile=certifi.where())


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    """Скачивает файл, если его ещё нет. Возвращает True, если скачал."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        last_error = None
        for context in _ssl_contexts():
            try:
                with urllib.request.urlopen(url, context=context, timeout=60) as response, open(part, "wb") as f:
                    shutil.copyfileobj(response, f)
                part.replace(dest)
                return True
            except urllib.error.URLError as e:
                if isinstance(e.reason, ssl.SSLCertVerificationError):
                    continue
                last_error = e
                break
            except Exception as e:
                last_error = e
                break
        else:
            raise SystemExit(CERT_HINT)
        if attempt == retries:
            raise last_error
        time.sleep(2 * attempt)
    return False


def ensure_annotations() -> Path:
    ann_path = COCO_DIR / "annotations" / "instances_val2017.json"
    if ann_path.exists():
        return ann_path
    zip_path = COCO_DIR / "annotations_trainval2017.zip"
    print("Скачиваю аннотации COCO (архив ~240 МБ, нужен один раз)...")
    fetch(ANN_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(ANN_MEMBER) as src, open(ann_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()  # архив больше не нужен
    return ann_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500, help="сколько изображений скачать (по умолчанию 500)")
    args = parser.parse_args()

    ann_path = ensure_annotations()
    images = json.loads(ann_path.read_text(encoding="utf-8"))["images"]
    sample = random.Random(SEED).sample(images, min(args.n, len(images)))
    print(f"Скачиваю {len(sample)} изображений val2017...")
    downloaded = 0
    for i, img in enumerate(sample, 1):
        if fetch(f"{BASE_URL}/val2017/{img['file_name']}", COCO_DIR / "val2017" / img["file_name"]):
            downloaded += 1
        if i % 50 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)}")
    print(f"Готово. Скачано новых: {downloaded}. Папка: {COCO_DIR}")


if __name__ == "__main__":
    main()
