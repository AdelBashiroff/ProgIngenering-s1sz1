"""Content-based, item-based collaborative, hybrid и две эвристики для MovieLens."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

GENRES = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
METHODS = ('popular', 'bayesian', 'content', 'item_cf', 'hybrid')
DATA = Path(__file__).resolve().parent / 'data'


def load_data(data_dir=DATA):
    ratings = pd.read_csv(data_dir / 'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    items = pd.read_csv(data_dir / 'u.item', sep='|', encoding='latin-1',
                        names=['item_id', 'title', 'release_date', 'video_date', 'url'] + GENRES)
    if ratings.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Повторные пары пользователь-фильм')
    return ratings, items


def temporal_split(ratings, test_fraction=.2):
    """Последние 20% оценок каждого пользователя; одинаковое время не разделяется."""
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction должен быть между 0 и 1')
    train, test = [], []
    for _, group in ratings.groupby('user_id', sort=True):
        group = group.sort_values(['timestamp', 'item_id'], kind='stable')
        if len(group) < 2:
            train.append(group)
            continue
        count = max(1, int(np.ceil(len(group) * test_fraction)))
        cutoff = group.iloc[-count]['timestamp']
        before, after = group[group.timestamp < cutoff], group[group.timestamp >= cutoff]
        # При единственном времени у пользователя нет честной временной истории.
        if before.empty:
            train.append(group)
        else:
            train.append(before)
            test.append(after)
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True) if test else ratings.iloc[:0].copy()


class Recommender:
    def __init__(self, train, items, neighbors=40):
        if train.empty or items.empty:
            raise ValueError('Нужны непустые train и каталог')
        self.items = items.sort_values('item_id').reset_index(drop=True).copy()
        self.ids = self.items.item_id.to_numpy()
        self.index = {int(v): i for i, v in enumerate(self.ids)}
        self.users = sorted(train.user_id.unique())
        self.user_index = {int(v): i for i, v in enumerate(self.users)}
        self.train = train.copy()
        self.seen = {int(u): set(g.item_id) for u, g in train.groupby('user_id')}
        positives = train[train.rating >= 4]
        rows = positives.user_id.map(self.user_index).to_numpy()
        cols = positives.item_id.map(self.index).to_numpy()
        self.likes = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.users), len(self.ids)))
        self.popular = np.asarray(self.likes.sum(axis=0)).ravel()
        stats = train.groupby('item_id').rating.agg(['count', 'mean']).reindex(self.ids)
        counts = stats['count'].fillna(0).to_numpy()
        means = stats['mean'].fillna(train.rating.mean()).to_numpy()
        self.bayesian = (counts * means + 20 * train.rating.mean()) / (counts + 20)
        self.bayesian[counts == 0] = 0
        self.features = TfidfTransformer().fit_transform(self.items[GENRES].to_numpy())
        similarities = cosine_similarity(self.likes.T, dense_output=True)
        np.fill_diagonal(similarities, 0)
        # Для каждого исходного фильма оставляем ближайших соседей.
        n = min(neighbors, len(self.ids) - 1)
        if n < 1:
            similarities[:] = 0
        else:
            order = np.argsort(-similarities, axis=1, kind='stable')
            mask = np.zeros_like(similarities, dtype=bool)
            np.put_along_axis(mask, order[:, :n], True, axis=1)
            similarities[~mask] = 0
        self.item_sim = csr_matrix(similarities)

    def _personal(self, user_id):
        row = self.user_index.get(user_id)
        if row is None or self.likes[row].nnz == 0:
            return None, None
        likes = self.likes[row]
        profile = normalize(likes @ self.features)
        content = (profile @ self.features.T).toarray().ravel()
        collaborative = (likes @ self.item_sim).toarray().ravel()
        return content, collaborative

    def recommend(self, user_id, method='item_cf', k=10):
        if method not in METHODS:
            raise ValueError(f'Неизвестный метод: {method}')
        if k < 1:
            raise ValueError('k должен быть положительным')
        if method in ('popular', 'bayesian'):
            scores = getattr(self, method).copy()
        else:
            content, cf = self._personal(user_id)
            if content is None:
                scores = self.popular.copy()
            elif method == 'content':
                scores = content
            elif method == 'item_cf':
                scores = cf
            else:
                scores = .5 * content / max(content.max(), 1e-12) + .5 * cf / max(cf.max(), 1e-12)
        excluded = self.seen.get(user_id, set())
        # Детерминированные ничьи: популярность, затем item_id. Все ранее оценённые
        # фильмы исключаются, в том числе с оценкой ниже 4.
        order = np.lexsort((self.ids, -self.popular, -scores))
        return [int(self.ids[i]) for i in order if int(self.ids[i]) not in excluded][:k]

    def titles(self, ids):
        by_id = self.items.set_index('item_id')['title']
        return [{'item_id': int(i), 'title': by_id.loc[i]} for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', type=int, default=1)
    parser.add_argument('--method', choices=METHODS, default='item_cf')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    ratings, items = load_data()
    train, _ = temporal_split(ratings)
    model = Recommender(train, items)
    print(json.dumps(model.titles(model.recommend(args.user, args.method, args.k)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
