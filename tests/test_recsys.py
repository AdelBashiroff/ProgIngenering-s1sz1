import numpy as np
import pandas as pd
import pytest
from recsys.recommender import GENRES, METHODS, Recommender, temporal_split
from recsys.evaluate import ranking_metrics


@pytest.fixture
def data():
    items = pd.DataFrame({'item_id': [1,2,3,4,5], 'title': ['A','B','C','D','E']})
    for g in GENRES:
        items[g] = 0
    items.loc[[0,1,4], 'Comedy'] = 1
    items.loc[[2,3], 'Drama'] = 1
    ratings = pd.DataFrame([(1,1,5,1),(1,3,1,2),(2,1,5,1),(2,2,5,2),(2,4,2,3),
                            (3,2,4,1),(3,3,5,2)],columns=['user_id','item_id','rating','timestamp'])
    return ratings, items


def test_metrics_known_ranking():
    r = ranking_metrics([2,1,3], {1,2}, k=3)
    assert r['precision_at_10'] == pytest.approx(2/3)
    assert r['recall_at_10'] == 1
    assert r['ndcg_at_10'] == 1
    assert r['map_at_10'] == 1
    assert ranking_metrics([8,9], {1}, k=2)['ndcg_at_10'] == 0


def test_temporal_no_overlap(data):
    ratings, _ = data
    train, test = temporal_split(ratings)
    assert len(train) + len(test) == len(ratings)
    assert not set(map(tuple,train[['user_id','item_id']].values)) & set(map(tuple,test[['user_id','item_id']].values))
    for u, group in test.groupby('user_id'):
        assert train[train.user_id == u].timestamp.max() < group.timestamp.min()


def test_timestamp_ties():
    df = pd.DataFrame([(1,1,5,1),(1,2,5,2),(1,3,5,2)],columns=['user_id','item_id','rating','timestamp'])
    train, test = temporal_split(df)
    assert train.item_id.tolist() == [1]
    assert set(test.item_id) == {2,3}


@pytest.mark.parametrize('method', METHODS)
def test_excludes_all_seen_and_unique(data, method):
    ratings, items = data
    model = Recommender(ratings, items, neighbors=2)
    result = model.recommend(1, method, k=10)
    assert not {1,3} & set(result)  # исключается и негативная оценка
    assert len(result) == len(set(result)) == 3
    assert set(result).issubset(set(items.item_id))


def test_cold_user_fallback(data):
    model = Recommender(*data)
    assert model.recommend(999,'item_cf') == model.recommend(999,'popular')
    assert model.recommend(999,'content') == model.recommend(999,'popular')


def test_content_recommends_same_genre(data):
    model = Recommender(*data)
    assert set(model.recommend(1, 'content', 2)) == {2,5}


def test_invalid_method_and_k(data):
    model = Recommender(*data)
    with pytest.raises(ValueError):model.recommend(1, 'missing')
    with pytest.raises(ValueError):model.recommend(1, k=0)
