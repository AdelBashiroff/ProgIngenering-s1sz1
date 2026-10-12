"""Сбор воспроизводимого набора: python -m image.capture_screenshots.

Манифест хранит URL, HTTP-статус, время, SHA256, ошибки и ручное решение QA.
--limit не стирает ранее собранные записи; кеш проверяется по URL и SHA256.
"""
import argparse
import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

try:
    from .labels import LABELS, normalize_label
except ImportError:
    from labels import LABELS, normalize_label

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
VIEWPORT = {'width': 1440, 'height': 900}
COOKIE_BUTTONS = ['Reject all', 'Reject All', 'Accept all', 'Accept All', 'Accept',
                  'I agree', 'Got it', 'Принять все', 'Принять', 'Хорошо', 'Понятно']


def load_urls(path):
    df = pd.read_csv(path, encoding='utf-8-sig').fillna('')
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {'url', 'label'}.issubset(df.columns):
        raise ValueError('Требуются колонки url и label')
    df['url'] = df['url'].str.strip()
    df['label'] = df['label'].map(normalize_label)
    if df['label'].isna().any():
        raise ValueError(f'Допустимые метки: {LABELS}')
    for url in df['url']:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError(f'Некорректный URL: {url}')
    return df.drop_duplicates('url').reset_index(drop=True)


def image_key(url, label):
    return f'{label}/{hashlib.sha256(url.encode()).hexdigest()[:16]}.jpg'


async def capture(df, args):
    from playwright.async_api import async_playwright
    manifest_path = args.data_dir / 'manifest.csv'
    previous = pd.read_csv(manifest_path).fillna('').to_dict('records') if manifest_path.exists() else []
    records = {r['url']: r for r in previous}
    semaphore = asyncio.Semaphore(args.workers)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale=args.locale)

        async def one(row):
            async with semaphore:
                url, label = row['url'], row['label']
                rel = image_key(url, label)
                path = args.data_dir / 'images' / rel
                old = records.get(url, {})
                if path.exists() and not args.force and old.get('sha256') == hashlib.sha256(path.read_bytes()).hexdigest():
                    print(f'cached {url}', flush=True)
                    return
                record = {**row, 'file': rel, 'final_url': url, 'status': 'error',
                          'http_status': '', 'title': '', 'warning': '', 'error': '',
                          'captured_at': datetime.now(timezone.utc).isoformat(),
                          'sha256': '', 'qa': 'pending', 'qa_note': ''}
                page = await context.new_page()
                try:
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=args.timeout * 1000)
                    record['http_status'] = response.status if response else ''
                    if response and response.status >= 400:
                        raise ValueError(f'HTTP {response.status}')
                    await page.wait_for_timeout(args.wait * 1000)
                    for text in COOKIE_BUTTONS:
                        button = page.get_by_role('button', name=text, exact=True).first
                        try:
                            if await button.is_visible():
                                await button.click(timeout=700)
                                break
                        except Exception:
                            pass
                    await page.wait_for_timeout(500)
                    record['title'] = await page.title()
                    record['final_url'] = page.url
                    body = (await page.locator('body').inner_text(timeout=2000))[:2000].lower()
                    if any(s in (record['title'] + ' ' + body).lower() for s in [
                        'just a moment', 'verify you are human', 'access denied', 'доступ ограничен',
                        'подтвердите, что вы не робот', 'checking your browser']):
                        record['warning'] = 'Возможная антибот-страница; нужна ручная проверка'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    await page.screenshot(path=str(path), type='jpeg', quality=85)
                    record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    record['status'] = 'ok'
                    if path.stat().st_size < 20000:
                        record['warning'] += ' Маленький файл; нужна ручная проверка'
                except Exception as exc:
                    record['error'] = str(exc).splitlines()[0][:250]
                finally:
                    await page.close()
                records[url] = record
                pd.DataFrame(records.values()).to_csv(manifest_path, index=False)
                print(f"{record['status']}: {url} {record['warning']} {record['error']}", flush=True)
        try:
            await asyncio.gather(*(one(row) for row in df.to_dict('records')))
        finally:
            await context.close()
            await browser.close()
    return pd.DataFrame(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urls', type=Path, default=DATA_DIR / 'urls.csv')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--timeout', type=int, default=25)
    parser.add_argument('--wait', type=float, default=3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--locale', default='en-US')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.wait < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Проверьте положительные workers, timeout, limit и неотрицательный wait')
    args.data_dir.mkdir(parents=True, exist_ok=True)
    df = load_urls(args.urls)
    if args.limit:
        df = df.head(args.limit)
    result = asyncio.run(capture(df, args))
    print(result.groupby(['label', 'status']).size())
    print('Просмотрите скриншоты и заполните qa=accept/reject и qa_note в manifest.csv.')


if __name__ == '__main__':
    main()
