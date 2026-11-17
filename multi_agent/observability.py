"""Локальные OTel-трейсы, траектории, метрики Prometheus и алерты."""
import json
import time
from contextlib import contextmanager
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


class JsonSpanExporter(SpanExporter):
    def __init__(self, path):
        self.path = path

    def export(self, spans):
        for span in spans:
            append_json(self.path, json.loads(span.to_json()))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class Observability:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(
            JsonSpanExporter(self.directory / 'spans.jsonl')))
        self.tracer = self.provider.get_tracer('movie-agent')
        self.registry = CollectorRegistry()
        self.runs = Counter('agent_runs', 'Запуски по исходу', ['status'], registry=self.registry)
        self.duration = Histogram('agent_run_seconds', 'Время полного запуска',
                                  buckets=(5, 15, 30, 60, 120, 300, 600), registry=self.registry)
        self.llm_time = Histogram('agent_llm_seconds', 'Время LLM включая загрузку', ['role'],
                                  buckets=(1, 5, 15, 30, 60, 120, 300), registry=self.registry)
        self.tokens = Counter('agent_llm_tokens', 'Токены LLM', ['role', 'direction'], registry=self.registry)
        self.tool_calls = Counter('agent_tool_calls', 'Вызовы инструментов', ['tool', 'cached'], registry=self.registry)
        self.rejections = Counter('agent_validation_rejections', 'Отклонённые ответы', registry=self.registry)

    @contextmanager
    def span(self, name, **attributes):
        with self.tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    def event(self, run_id, node, **fields):
        append_json(self.directory / 'trajectories.jsonl',
                    {'time': time.time(), 'run_id': run_id, 'node': node, **fields})

    def llm(self, role, result):
        self.llm_time.labels(role).observe(result['latency_s'])
        self.tokens.labels(role, 'input').inc(result.get('prompt_tokens', 0))
        self.tokens.labels(role, 'output').inc(result.get('completion_tokens', 0))

    def finish(self, run_id, status, elapsed):
        self.runs.labels(status).inc()
        self.duration.observe(elapsed)
        self.event(run_id, 'finish', status=status, elapsed_s=elapsed)
        alerts = []
        if status in ('error', 'rejected'):
            alerts.append({'rule': 'AgentFailed', 'severity': 'warning', 'status': status})
        if elapsed > 120:
            alerts.append({'rule': 'AgentSlow', 'severity': 'warning', 'seconds': elapsed})
        for alert in alerts:
            append_json(self.directory / 'alerts.jsonl', {'time': time.time(), 'run_id': run_id, **alert})
        self.save_metrics()

    def save_metrics(self):
        (self.directory / 'metrics.prom').write_bytes(generate_latest(self.registry))

    def close(self):
        self.provider.shutdown()
