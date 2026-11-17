import pytest

from multi_agent.constraints import GENRE_PATTERNS, normalize_plan
from multi_agent.schemas import Plan
from multi_agent.tools import MovieTools
from recsys.recommender import GENRES


def dirty_plan(**changes):
    data = {'tool': 'recommend', 'method': 'item_cf',
            'genres': ['Comedy', "Children's", 'Western'], 'avoid': [],
            'query': '', 'k': 1}
    return Plan(**(data | changes))


def test_explicit_request_replaces_hallucinated_constraints():
    original = dirty_plan()
    fixed = normalize_plan(original, 'Посоветуй 3 комедии, без ужасов.')
    assert fixed.genres == ['Comedy']
    assert fixed.avoid == ['Horror']
    assert fixed.k == 3
    assert fixed.tool == original.tool and fixed.method == original.method
    assert original.genres == ['Comedy', "Children's", 'Western']


@pytest.mark.parametrize('phrase', ['без ужасов', 'кроме ужасов', 'исключи ужасы',
                                    'не хочу ужасы', 'не предлагай ужасы', 'не люблю ужасы'])
def test_common_ban_forms(phrase):
    fixed = normalize_plan(dirty_plan(), 'Покажи комедию, ' + phrase)
    assert fixed.genres == ['Comedy'] and fixed.avoid == ['Horror']


def test_negation_list_and_new_positive_clause():
    fixed = normalize_plan(dirty_plan(), 'Без ужасов и триллеров, но две драмы.')
    assert fixed.avoid == ['Horror', 'Thriller']
    assert fixed.genres == ['Drama'] and fixed.k == 2


def test_explicit_conflict_reaches_tools_as_empty_result():
    fixed = normalize_plan(dirty_plan(), 'Хочу комедию, но без комедий.')
    assert fixed.genres == fixed.avoid == ['Comedy']
    # Конфликт должен остановиться до загрузки каталога или модели.
    tools = MovieTools.__new__(MovieTools)
    evidence = tools.call(fixed, 1, {'genres': [], 'avoid': []})
    assert evidence['items'] == []


@pytest.mark.parametrize(('word', 'expected'), [('один', 1), ('одну', 1), ('две', 2),
                                              ('три', 3), ('четыре', 4), ('пять', 5)])
def test_russian_counts(word, expected):
    assert normalize_plan(dirty_plan(), f'Покажи {word} фильма.').k == expected


def test_number_of_methods_does_not_replace_number_of_movies():
    fixed = normalize_plan(dirty_plan(tool='compare'), 'Сравни две модели и выбери три фильма.')
    assert fixed.k == 3 and fixed.tool == 'compare'


def test_no_explicit_genres_leaves_profile_to_tools():
    fixed = normalize_plan(dirty_plan(avoid=['Horror']), 'Покажи 2 фильма из моей истории.')
    assert fixed.genres == fixed.avoid == []
    assert fixed.k == 2


def test_search_title_does_not_become_genre_or_count_constraint():
    original = dirty_plan(tool='search', query='Horror 3')
    assert normalize_plan(original, 'Найди Horror 3, фильм с таким названием.') == original


def test_english_genres_and_all_movielens_labels():
    assert set(GENRE_PATTERNS) == set(GENRES)
    fixed = normalize_plan(dirty_plan(), 'Recommend five science fiction films without horror.')
    assert fixed.genres == ['Sci-Fi'] and fixed.avoid == ['Horror']


@pytest.mark.parametrize(('word', 'genre'), [
    ('боевики', 'Action'), ('приключения', 'Adventure'), ('мультфильмы', 'Animation'),
    ('детские', "Children's"), ('криминальные', 'Crime'), ('документальные', 'Documentary'),
    ('фэнтези', 'Fantasy'), ('нуар', 'Film-Noir'), ('мюзиклы', 'Musical'),
    ('детективы', 'Mystery'), ('романтические', 'Romance'), ('фантастика', 'Sci-Fi'),
    ('военные', 'War'), ('вестерны', 'Western'), ('неизвестный жанр', 'unknown')])
def test_catalog_vocabulary(word, genre):
    assert normalize_plan(dirty_plan(), f'Хочу {word}').genres == [genre]
