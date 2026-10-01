"""Скачивает подмножество ESC-10 (400 записей, ~175 МБ) из репозитория ESC-50.

Запуск:
    python download_esc10.py              # все 400 файлов
    python download_esc10.py --per-class 5   # быстрая проверка: по 5 файлов на класс

Файлы сохраняются в data/ESC-50/ (папка не должна попадать в Git — см. README).
"""
import argparse
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
DATA_DIR = Path(__file__).parent / "data" / "ESC-50"


def _ssl_contexts():
    """Сначала системные сертификаты, затем (запасной вариант) сертификаты из certifi.

    На macOS с Python с python.org системных сертификатов может не быть — тогда выручает certifi.
    """
    yield ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return
    yield ssl.create_default_context(cafile=certifi.where())


CERT_HINT = (
    "Ошибка SSL-сертификата. На macOS запустите «/Applications/Python 3.x/Install Certificates.command» "
    "или выполните: pip install certifi"
)


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
                    continue  # пробуем следующий набор сертификатов
                last_error = e
                break
            except Exception as e:
                last_error = e
                break
        else:
            raise SystemExit(CERT_HINT)  # сертификаты не подошли ни в одном варианте
        if attempt == retries:
            raise last_error
        time.sleep(2 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=None, help="сколько файлов на класс (по умолчанию все)")
    args = parser.parse_args()

    meta_path = DATA_DIR / "meta" / "esc50.csv"
    fetch(f"{BASE_URL}/meta/esc50.csv", meta_path)
    meta = pd.read_csv(meta_path)
    meta = meta[meta["esc10"]]
    if args.per_class:
        meta = meta.groupby("category").head(args.per_class)

    files = meta["filename"].tolist()
    print(f"Нужно файлов: {len(files)}")
    downloaded = 0
    for n, name in enumerate(files, 1):
        if fetch(f"{BASE_URL}/audio/{name}", DATA_DIR / "audio" / name):
            downloaded += 1
        if n % 20 == 0 or n == len(files):
            print(f"  {n}/{len(files)}")
    print(f"Готово. Скачано новых: {downloaded}. Папка: {DATA_DIR}")


if __name__ == "__main__":
    main()
