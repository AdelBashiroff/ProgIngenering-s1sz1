"""python -m multi_agent.runtime.setup — около 4.2 GB, без установки в систему."""

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected_hash:
        print("Уже есть:", path.name, flush=True)
        return
    temporary = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "-L", "--fail", "--retry", "2", "--output", str(temporary), url], check=True)
    actual = sha256(temporary)
    if actual != expected_hash:
        raise RuntimeError(f"SHA256 не совпал для {path.name}: {actual}")
    temporary.replace(path)
    print("Проверен SHA256:", path.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-engine", action="store_true", help="Загрузить только веса, движок указан через LLAMA_SERVER")
    args = parser.parse_args()
    config = json.loads((ROOT / "multi_agent/models.json").read_text())
    engine = config["engine"]
    if not args.verify_only and not args.skip_engine:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Этот архив — для macOS ARM64. Укажите свой llama-server через LLAMA_SERVER.")
        archive = ROOT / ".cache/engine" / Path(engine["archive_url"]).name
        download(engine["archive_url"], archive, engine["archive_sha256"])
        with tarfile.open(archive) as bundle:
            # В архиве только обычные файлы/ссылки внутри каталога релиза.
            for member in bundle.getmembers():
                target = (archive.parent / member.name).resolve()
                if not target.is_relative_to(archive.parent.resolve()):
                    raise RuntimeError("Недопустимый путь в архиве")
            bundle.extractall(archive.parent, filter="data")
    for model in config["models"].values():
        weights = ROOT / ".cache/models" / model["filename"]
        if args.verify_only:
            if not weights.exists() or sha256(weights) != model["sha256"]:
                raise RuntimeError(f"Нет проверенных весов: {weights.name}")
            print("SHA256 OK:", weights.name)
        else:
            url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{model["filename"]}'
            download(url, weights, model["sha256"])


if __name__ == "__main__":
    main()
