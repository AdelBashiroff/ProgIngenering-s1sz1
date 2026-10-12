"""Скачать MovieLens 100K из GroupLens и извлечь только необходимые файлы."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent / 'data'
URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    archive = BASE / 'ml-100k.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        urlretrieve(URL, temporary)
        temporary.replace(archive)
    with ZipFile(archive) as z:
        for name in ['u.data', 'u.item', 'u.genre', 'README']:
            (BASE / name).write_bytes(z.read('ml-100k/' + name))
    metadata = {'url': URL, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'dataset': 'MovieLens 100K', 'ratings': 100000}
    (BASE.parent / 'results' / 'data_source.json').write_text(json.dumps(metadata, indent=2))
    print(BASE)


if __name__ == '__main__':
    main()
