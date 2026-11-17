from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from recsys.recommender import GENRES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Plan(StrictModel):
    tool: Literal['recommend', 'history', 'search', 'compare', 'refuse']
    method: Literal['item_cf', 'lightgcn', 'content', 'hybrid', 'popular']
    genres: list[str] = Field(max_length=4)
    avoid: list[str] = Field(max_length=4)
    query: str = Field(max_length=120)
    k: int = Field(ge=1, le=5)

    def check_genres(self):
        if any(g not in GENRES for g in self.genres + self.avoid):
            raise ValueError('Жанр отсутствует в каталоге')


class Choice(StrictModel):
    item_id: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=220)


class Draft(StrictModel):
    choices: list[Choice] = Field(min_length=1, max_length=5)


class Verdict(StrictModel):
    approved: bool
    issues: list[str] = Field(max_length=6)


def check_draft(draft, evidence, k):
    """Проверки фактов, которые не стоит поручать языковой модели."""
    allowed = {row['item_id'] for row in evidence['items']}
    ids = [choice.item_id for choice in draft.choices]
    issues = []
    if len(ids) != len(set(ids)):
        issues.append('Повтор фильма')
    if len(ids) > k:
        issues.append('Превышено число фильмов')
    if len(ids) < min(k, len(allowed)):
        issues.append('Выбрано меньше фильмов, чем запрошено и доступно')
    if not set(ids).issubset(allowed):
        issues.append('Есть ID вне ответа инструмента')
    return issues
