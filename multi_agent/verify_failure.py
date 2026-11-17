"""Проверить наблюдаемость настоящей ошибки backend без подмены LLM-ответа."""
import argparse
import json
from pathlib import Path

from multi_agent.llm import ModelRouter
from multi_agent.monitor import snapshot
from multi_agent.workflow import HERE, MovieAssistant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE / 'var/fault')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE / 'models.json').read_text())
    config['engine']['path'] = 'multi_agent/runtime/intentionally-missing-llama-server'
    config_path = args.directory / 'fault-models.json'
    config_path.write_text(json.dumps(config))
    app = MovieAssistant(ModelRouter(config_path, work_dir=args.directory / 'engine'), args.directory)
    error = None
    try:
        app.ask('Посоветуй одну комедию', 1)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        app.close()
    alerts_path = args.directory / 'alerts.jsonl'
    alerts = [json.loads(s) for s in alerts_path.read_text().splitlines()] if alerts_path.exists() else []
    metrics = (args.directory / 'metrics.prom').read_text()
    result = {'experiment': 'Deliberately unavailable engine; actual ModelRouter failure',
              'expected_error': error, 'alert_fired': any(r['rule'] == 'AgentFailed' for r in alerts),
              'error_metric_present': 'agent_runs_total{status="error"} 1.0' in metrics,
              'snapshot': snapshot(args.directory),
              'note': 'Fault experiment is separate from six normal end-to-end scenarios'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not (error and result['alert_fired'] and result['error_metric_present']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
