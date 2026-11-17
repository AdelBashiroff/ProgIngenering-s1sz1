import numpy as np
import pandas as pd
import pytest
import torch

from recsys.lightgcn import (LightGCN, LightGCNRecommender, negative_samples,
                            normalized_graph, train_model)


def test_graph_normalization_and_propagation():
    graph = normalized_graph([0, 0, 1], [0, 1, 1], 2, 2)
    expected = torch.tensor([[0., 0., 2**-.5, .5], [0., 0., 0., 2**-.5],
                             [2**-.5, 0., 0., 0.], [.5, 2**-.5, 0., 0.]])
    torch.testing.assert_close(graph.to_dense(), expected)
    model = LightGCN(2, 2, graph, dimensions=1, layers=1)
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor([[1.], [2.], [3.], [4.]]))
    users, items = model()
    result = torch.cat([users, items])
    torch.testing.assert_close(result, (model.embedding.weight + expected @ model.embedding.weight) / 2)


def test_negative_samples_exclude_all_seen():
    seen = np.array([[True, True, False, False], [False, False, True, True]])
    users = np.tile([0, 1], 100)
    negatives = negative_samples(users, seen, np.random.default_rng(42))
    assert not seen[users, negatives].any()
    with pytest.raises(ValueError):
        negative_samples(np.array([0]), np.ones((1, 2), dtype=bool), np.random.default_rng(42))


def test_train_checkpoint_and_fallback(tmp_path):
    torch.set_num_threads(1)
    ratings = pd.DataFrame([(1, 1, 5), (1, 2, 1), (2, 1, 5), (2, 3, 4), (3, 2, 1)],
                           columns=['user_id', 'item_id', 'rating'])
    items = pd.DataFrame({'item_id': [1, 2, 3, 4], 'title': ['A', 'B', 'C', 'D']})
    model, recommender, history, _, _ = train_model(ratings, items, epochs=10, dimensions=4)
    assert history[-1]['loss'] < history[0]['loss']
    assert recommender.recommend(1, 2) == [3, 4]
    assert recommender.recommend(999, 2) == [1, 3]
    assert recommender.recommend(3, 2) == [1, 3]  # только низкие оценки, нет рёбер
    path = tmp_path / 'model.pth'
    recommender.save(path, model.state_dict())
    loaded = LightGCNRecommender.load(path)
    assert loaded.recommend(1) == recommender.recommend(1)
    assert loaded.titles([3]) == [{'item_id': 3, 'title': 'C'}]
    with pytest.raises(ValueError):
        loaded.recommend(1, k=0)
