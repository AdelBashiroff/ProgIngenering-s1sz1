import pytest
import json
from pathlib import Path

from multi_agent.memory import Memory
from multi_agent.schemas import Draft, check_draft
from multi_agent.workflow import MovieAssistant
from multi_agent.llm import ModelRouter
from multi_agent.schemas import Plan
from multi_agent.tools import MovieTools


def test_memory_expiry_revision_and_forget(tmp_path):
    clock = [100.0]
    memory = Memory(tmp_path / 'db.sqlite3', clock=lambda: clock[0])
    plan = {'method': 'item_cf'}
    first = memory.cache_key(1, plan, 'data-v1')
    memory.cache(first, {'items': [1]}, ttl=10)
    assert memory.cached(first) == {'items': [1]}
    clock[0] += 11
    assert memory.cached(first) is None
    memory.update_profile(1, ['Comedy'], ['Horror'], ttl_days=1)
    assert memory.cache_key(1, plan, 'data-v1') != first
    assert memory.cache_key(1, plan, 'data-v2') != memory.cache_key(1, plan, 'data-v1')
    assert memory.profile(2)['genres'] == []
    clock[0] += 86401
    assert memory.profile(1)['genres'] == []
    memory.update_profile(1, ['Drama'], [])
    memory.record_run('abc', 1, 'approved')
    memory.forget(1)
    assert memory.profile(1)['genres'] == []
    assert memory.db.execute('SELECT count(*) FROM runs').fetchone()[0] == 0
    with pytest.raises(ValueError):
        memory.update_profile(1, ['Comedy'], ['Comedy'])
    memory.close()


def test_rules_reject_unknown_and_duplicate_ids():
    draft = Draft(choices=[{'item_id': 99, 'reason': 'Comedy'}, {'item_id': 99, 'reason': 'Comedy'}])
    issues = check_draft(draft, {'items': [{'item_id': 1}]}, k=1)
    assert len(issues) == 3


def test_router_rejects_same_family_validator(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / 'multi_agent/models.json').read_text())
    config['roles']['validator'] = config['roles']['writer']
    path = tmp_path / 'models.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='семейство'):
        ModelRouter(path, work_dir=tmp_path)


def test_explicit_genre_conflict_returns_no_results():
    # Конфликт должен завершиться до обращения к модели, а не снять фильтр жанров.
    tools = MovieTools.__new__(MovieTools)
    plan = Plan(tool='recommend', method='item_cf', genres=['Horror'],
                avoid=[], query='', k=3)
    assert tools.call(plan, 1, {'genres': [], 'avoid': ['Horror']})['items'] == []


def test_cold_start_evidence_names_actual_fallback():
    import pandas as pd
    from types import SimpleNamespace
    from recsys.recommender import GENRES
    tools = MovieTools.__new__(MovieTools)
    tools.model = SimpleNamespace(user_index={}, seen={})
    tools.catalog = pd.DataFrame([{'item_id': 1, 'title': 'Fixture',
                                   **{g: int(g == 'Comedy') for g in GENRES}}]).set_index('item_id')
    tools.version = 'fixture'
    tools._recommend = lambda user, method, count: [1]
    plan = Plan(tool='recommend', method='item_cf', genres=[], avoid=[], query='', k=3)
    result = tools.call(plan, 99999, {'genres': [], 'avoid': []})
    assert result['method'] == 'popular'
    assert result['fallback_to_popular'] is True
    assert result['items']


class FakeTools:
    version = 'fixture-v1'

    def call(self, plan, user_id, profile):
        return {'items': [{'item_id': 1, 'title': 'Fixture', 'genres': ['Comedy']}],
                'method': plan.method}


class ScriptedRouter:
    """Только unit tests: проверяет переходы графа без загрузки LLM."""
    def __init__(self, judge_approves):
        self.judge_approves = judge_approves
        self.calls = []

    def generate(self, role, system, payload, schema):
        self.calls.append(role)
        data = {'planner': {'tool': 'recommend', 'method': 'item_cf', 'genres': [],
                            'avoid': [], 'query': '', 'k': 1},
                'writer': {'choices': [{'item_id': 1, 'reason': 'Комедия по жанру.'}]},
                'validator': {'approved': self.judge_approves,
                              'issues': [] if self.judge_approves else ['Недостаточно данных']}}[role]
        return {'data': data, 'model': 'test-double', 'family': role, 'latency_s': .001}

    def close(self):
        pass


@pytest.mark.parametrize('approve', [True, False])
def test_graph_requires_judge_and_bounds_retry(tmp_path, approve):
    router = ScriptedRouter(approve)
    app = MovieAssistant(router, tmp_path, tools=FakeTools())
    result = app.ask('Покажи комедию', 1)
    assert result['status'] == ('approved' if approve else 'rejected')
    assert bool(result['answer']) == approve
    assert router.calls == ['planner', 'writer', 'validator'] * 1 + ([] if approve else ['writer', 'validator'])
    assert result['attempts'] == (1 if approve else 2)
    assert (tmp_path / 'spans.jsonl').exists()
    assert 'agent_runs_total' in (tmp_path / 'metrics.prom').read_text()
    if not approve:
        assert 'AgentFailed' in (tmp_path / 'alerts.jsonl').read_text()
    app.close()
