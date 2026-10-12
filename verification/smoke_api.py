"""Настоящий HTTP-запрос к запущенному серверу: python verification/smoke_api.py IMAGE."""
import argparse
import json
from pathlib import Path
import time
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    with httpx.Client(timeout=300) as client:
        health = client.get(args.url + '/health')
        health.raise_for_status()
        start = time.perf_counter()
        with args.image.open('rb') as f:
            response = client.post(args.url + '/predict', files={'file': (args.image.name, f, 'image/jpeg')})
        response.raise_for_status()
        result = response.json()
        assert result['label'] in {'landing', 'shop', 'blog'}
        assert 0 <= result['score'] <= 1
        assert abs(sum(result['probabilities'].values()) - 1) < .001
        record = {'health': health.json(), 'status_code': response.status_code,
                  'image': str(args.image), 'elapsed_s': round(time.perf_counter() - start, 3),
                  'prediction': result}
        Path('verification/api_real_request.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
