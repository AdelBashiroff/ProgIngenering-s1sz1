"""Локальный exporter и наблюдение за завершёнными запусками без повторных LLM-вызовов."""
import argparse
import json
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

from multi_agent.observability import append_json
from multi_agent.workflow import HERE


def snapshot(directory):
    spans = directory / 'spans.jsonl'
    metrics = directory / 'metrics.prom'
    alerts = directory / 'alerts.jsonl'
    trajectory = directory / 'trajectories.jsonl'
    count = lambda p: sum(1 for _ in p.open()) if p.exists() else 0
    latest_age = time.time() - trajectory.stat().st_mtime if trajectory.exists() else None
    return {'time': time.time(), 'spans': count(spans), 'trajectory_events': count(trajectory),
            'alerts': count(alerts), 'metrics_present': metrics.exists(),
            'last_activity_age_s': latest_age, 'disk_free_mb': shutil.disk_usage(directory).free / 1024**2,
            'healthy': metrics.exists() and latest_age is not None and latest_age < 600}


def prune(directory, days):
    cutoff = time.time() - days * 86400
    # Spans use ISO timestamps; остальные JSONL используют epoch seconds.
    from datetime import datetime
    for name in ('spans.jsonl', 'trajectories.jsonl', 'alerts.jsonl', 'monitor.jsonl'):
        path = directory / name
        if not path.exists():
            continue
        kept = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            stamp = row.get('time')
            if stamp is None:
                stamp = datetime.fromisoformat(row['start_time'].replace('Z', '+00:00')).timestamp()
            if stamp >= cutoff:
                kept.append(line)
        path.write_text('\n'.join(kept) + ('\n' if kept else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE / 'var')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--port', type=int, default=9098)
    parser.add_argument('--retention-days', type=int, default=7)
    args = parser.parse_args()
    if args.interval < 1 or not 1 <= args.retention_days <= 365:
        parser.error('Неверный интервал или срок хранения')
    args.directory.mkdir(parents=True, exist_ok=True)
    if not args.once:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/metrics':
                    self.send_error(404)
                    return
                path = args.directory / 'metrics.prom'
                content = path.read_bytes() if path.exists() else b''
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                self.wfile.write(content)
        server = HTTPServer(('127.0.0.1', args.port), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
    while True:
        prune(args.directory, args.retention_days)
        result = snapshot(args.directory)
        append_json(args.directory / 'monitor.jsonl', result)
        if not result['healthy']:
            append_json(args.directory / 'alerts.jsonl', {'time': time.time(),
                        'rule': 'AgentTelemetryStale', 'severity': 'warning'})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
