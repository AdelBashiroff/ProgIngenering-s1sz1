import json
import time
import uuid
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from multi_agent.memory import Memory
from multi_agent.constraints import normalize_plan
from multi_agent.observability import Observability
from multi_agent.schemas import Draft, Plan, Verdict, check_draft
from multi_agent.tools import MovieTools
from recsys.recommender import GENRES

HERE = Path(__file__).resolve().parent


class State(TypedDict, total=False):
    run_id: str
    user_id: int
    request: str
    profile: dict
    plan: dict
    evidence: dict
    draft: dict
    issues: list[str]
    attempts: int
    status: str
    answer: list[dict]


class MovieAssistant:
    def __init__(self, router, directory=HERE / 'var', tools=None):
        self.router = router
        self.directory = Path(directory)
        self.memory = Memory(self.directory / 'memory.sqlite3')
        self.obs = Observability(self.directory)
        self.tools = tools or MovieTools()
        graph = StateGraph(State)
        graph.add_node('planner', self.plan)
        graph.add_node('tools', self.use_tools)
        graph.add_node('writer', self.write)
        graph.add_node('validator', self.validate)
        graph.add_node('finish', self.finish)
        graph.add_edge(START, 'planner')
        graph.add_edge('planner', 'tools')
        graph.add_conditional_edges('tools', lambda s: 'writer' if s['evidence']['items'] else 'finish')
        graph.add_edge('writer', 'validator')
        graph.add_conditional_edges('validator', self.next_step)
        graph.add_edge('finish', END)
        self.graph = graph.compile()

    def llm(self, state, role, payload, schema):
        prompt = (HERE / 'prompts' / f'{role}.md').read_text()
        if role == 'planner':
            prompt += '\n' + (HERE / 'SOUL.md').read_text()
        if role == 'writer':
            name = {'recommend': 'recommend', 'compare': 'compare'}.get(state['plan']['tool'], 'catalog')
            prompt += '\n' + (HERE / 'skills' / name / 'SKILL.md').read_text()
        with self.obs.span(role, run_id=state['run_id']) as span:
            result = self.router.generate(role, prompt, payload, schema.model_json_schema())
            self.obs.llm(role, result)
            span.set_attribute('llm.model', result['model'])
            span.set_attribute('llm.family', result['family'])
            span.set_attribute('llm.output_tokens', result.get('completion_tokens', 0))
            self.obs.event(state['run_id'], role, model=result['model'], family=result['family'],
                           latency_s=result['latency_s'], output=result['data'])
            return schema.model_validate(result['data'])

    def plan(self, state):
        result = self.llm(state, 'planner', {'request': state['request'],
                          'profile': state['profile'], 'catalog_genres': GENRES}, Plan)
        normalized = normalize_plan(result, state['request'])
        if normalized != result:
            self.obs.event(state['run_id'], 'constraint_guard', raw=result.model_dump(),
                           normalized=normalized.model_dump())
        result = normalized
        result.check_genres()
        return {'plan': result.model_dump()}

    def use_tools(self, state):
        plan = Plan.model_validate(state['plan'])
        with self.obs.span('tool.' + plan.tool, run_id=state['run_id']):
            key = self.memory.cache_key(state['user_id'], state['plan'], self.tools.version)
            evidence = self.memory.cached(key)
            cached = evidence is not None
            if not cached:
                evidence = self.tools.call(plan, state['user_id'], state['profile'])
                self.memory.cache(key, evidence)
            self.obs.tool_calls.labels(plan.tool, str(cached).lower()).inc()
            self.obs.event(state['run_id'], 'tools', tool=plan.tool, cached=cached,
                           evidence=evidence)
            return {'evidence': evidence}

    def write(self, state):
        draft = self.llm(state, 'writer', {'request': state['request'],
                         'evidence': state['evidence'], 'k': state['plan']['k'],
                         'previous_issues': state.get('issues', [])}, Draft)
        return {'draft': draft.model_dump(), 'attempts': state.get('attempts', 0) + 1}

    def validate(self, state):
        draft = Draft.model_validate(state['draft'])
        issues = check_draft(draft, state['evidence'], state['plan']['k'])
        # Судья вызывается и при ошибке ID: его оценка измеряется отдельно от правил.
        verdict = self.llm(state, 'validator', {'request': state['request'],
                           'k': state['plan']['k'], 'evidence': state['evidence'],
                           'draft': state['draft']}, Verdict)
        if not verdict.approved:
            issues.extend(verdict.issues or ['Независимый валидатор отклонил ответ'])
        elif verdict.issues:
            issues.extend(verdict.issues)
        if issues:
            self.obs.rejections.inc()
        self.obs.event(state['run_id'], 'validation', rules_pass=not check_draft(
            draft, state['evidence'], state['plan']['k']), judge_approved=verdict.approved,
            issues=issues, attempt=state['attempts'])
        return {'issues': issues, 'status': 'rejected' if issues else 'approved'}

    @staticmethod
    def next_step(state):
        return 'writer' if state['issues'] and state['attempts'] < 2 else 'finish'

    @staticmethod
    def finish(state):
        if not state['evidence']['items']:
            return {'status': 'no_results', 'answer': []}
        if state['status'] != 'approved':
            return {'answer': []}
        by_id = {row['item_id']: row for row in state['evidence']['items']}
        return {'answer': [{**by_id[c['item_id']], 'reason': c['reason']}
                           for c in state['draft']['choices']]}

    def ask(self, request, user_id=1):
        if not request.strip() or len(request) > 1500 or user_id < 1:
            raise ValueError('Нужен непустой запрос до 1500 символов и положительный user_id')
        run_id = uuid.uuid4().hex
        start = time.perf_counter()
        initial = {'run_id': run_id, 'user_id': user_id, 'request': request,
                   'profile': self.memory.profile(user_id), 'attempts': 0, 'issues': []}
        status = 'error'
        try:
            with self.obs.span('agent.run', run_id=run_id):
                self.obs.event(run_id, 'start', request=request, user_id=user_id)
                result = self.graph.invoke(initial, {'recursion_limit': 14})
            status = result['status']
            return {'run_id': run_id, 'status': status, 'answer': result['answer'],
                    'plan': result['plan'], 'issues': result.get('issues', []),
                    'note': result['evidence'].get('note', ''),
                    'attempts': result.get('attempts', 0),
                    'elapsed_s': round(time.perf_counter() - start, 3)}
        except Exception as exc:
            self.obs.event(run_id, 'error', error_type=type(exc).__name__, message=str(exc)[:500])
            raise
        finally:
            self.memory.record_run(run_id, user_id, status)
            self.obs.finish(run_id, status, time.perf_counter() - start)

    def close(self):
        self.memory.close()
        self.obs.close()
        self.router.close()
