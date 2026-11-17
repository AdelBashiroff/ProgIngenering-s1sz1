"""LightGCN: граф пользователь–фильм, BPR-обучение и загрузка рекомендаций."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / 'results' / 'advanced' / 'lightgcn.pth'


def normalized_graph(user_rows, item_cols, n_users, n_items):
    """D^(-1/2) A D^(-1/2), без self-loops; каждое ребро в обе стороны."""
    users = torch.as_tensor(user_rows, dtype=torch.long)
    items = torch.as_tensor(item_cols, dtype=torch.long) + n_users
    source, target = torch.cat([users, items]), torch.cat([items, users])
    size = n_users + n_items
    degree = torch.bincount(source, minlength=size).float()
    inverse = degree.clamp(min=1).rsqrt()
    weights = inverse[source] * inverse[target]
    return torch.sparse_coo_tensor(torch.stack([source, target]), weights,
                                   (size, size)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, graph, dimensions=32, layers=2):
        super().__init__()
        if dimensions < 1 or layers < 0:
            raise ValueError('dimensions > 0, layers >= 0')
        self.n_users = n_users
        self.layers = layers
        self.embedding = nn.Embedding(n_users + n_items, dimensions)
        nn.init.normal_(self.embedding.weight, std=.1)
        self.register_buffer('graph', graph)

    def forward(self):
        current = self.embedding.weight
        levels = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.graph, current)
            levels.append(current)
        result = torch.stack(levels).mean(dim=0)
        return result[:self.n_users], result[self.n_users:]

    def bpr_loss(self, users, positive, negative, regularization=1e-4):
        user_vectors, item_vectors = self()
        pos = (user_vectors[users] * item_vectors[positive]).sum(dim=1)
        neg = (user_vectors[users] * item_vectors[negative]).sum(dim=1)
        initial = self.embedding.weight
        penalty = (initial[users].square().sum() +
                   initial[self.n_users + positive].square().sum() +
                   initial[self.n_users + negative].square().sum()) / (2 * len(users))
        return F.softplus(neg - pos).mean() + regularization * penalty


def negative_samples(user_rows, seen_mask, rng):
    """Неоценённый train-фильм; явно низкие оценки также не выдаются за пропуск."""
    if np.any(seen_mask[user_rows].all(axis=1)):
        raise ValueError('Нет неоценённых фильмов для negative sampling')
    result = rng.integers(seen_mask.shape[1], size=len(user_rows))
    bad = seen_mask[user_rows, result]
    while bad.any():
        result[bad] = rng.integers(seen_mask.shape[1], size=int(bad.sum()))
        bad = seen_mask[user_rows, result]
    return result


class LightGCNRecommender:
    """Готовые embeddings: граф пересчитывается только при обучении."""
    def __init__(self, user_ids, item_ids, titles, user_vectors, item_vectors, seen, popular):
        self.users = np.asarray(user_ids, dtype=int)
        self.ids = np.asarray(item_ids, dtype=int)
        self.user_index = {int(u): row for row, u in enumerate(self.users)}
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        self.item_vectors = np.asarray(item_vectors, dtype=np.float32)
        self.seen = {int(u): set(map(int, ids)) for u, ids in seen.items()}
        self.popular = np.asarray(popular)
        self.title_by_id = dict(zip(map(int, self.ids), titles))

    @classmethod
    def from_model(cls, model, train, items):
        with torch.no_grad():
            users, films = model()
        user_ids = sorted(train.user_id.unique())
        sorted_items = items.sort_values('item_id')
        seen = {int(u): set(g.item_id) for u, g in train.groupby('user_id')}
        popular = train[train.rating >= 4].groupby('item_id').size().reindex(sorted_items.item_id, fill_value=0)
        return cls(user_ids, sorted_items.item_id, sorted_items.title.tolist(),
                   users.numpy(), films.numpy(), seen, popular.to_numpy())

    def recommend(self, user_id, k=10):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k должен быть положительным целым числом')
        row = self.user_index.get(user_id)
        # У пользователей без положительных train-событий в графе нет рёбер.
        if row is None or not np.any(self.user_vectors[row]):
            scores = self.popular.astype(float)
        else:
            scores = self.item_vectors @ self.user_vectors[row]
            scores[self.popular == 0] = -np.inf
        excluded = self.seen.get(user_id, set())
        order = np.lexsort((self.ids, -self.popular, -scores))
        return [int(self.ids[i]) for i in order if int(self.ids[i]) not in excluded][:k]

    def titles(self, ids):
        return [{'item_id': int(i), 'title': self.title_by_id[int(i)]} for i in ids]

    def save(self, path, state_dict=None, config=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'format_version': 1, 'user_ids': torch.tensor(self.users),
                   'item_ids': torch.tensor(self.ids), 'titles': list(self.title_by_id.values()),
                   'user_vectors': torch.tensor(self.user_vectors),
                   'item_vectors': torch.tensor(self.item_vectors),
                   'seen': {u: sorted(ids) for u, ids in self.seen.items()},
                   'popular': torch.tensor(self.popular), 'config': config or {},
                   'state_dict': state_dict or {}}
        torch.save(payload, path)

    @classmethod
    def load(cls, path=DEFAULT_CHECKPOINT):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != 1:
            raise ValueError('Неизвестная версия checkpoint')
        return cls(payload['user_ids'].numpy(), payload['item_ids'].numpy(), payload['titles'],
                   payload['user_vectors'].numpy(), payload['item_vectors'].numpy(),
                   payload['seen'], payload['popular'].numpy())


def train_model(train, items, dimensions=32, layers=2, epochs=150, learning_rate=.01,
                regularization=1e-4, seed=42, validation=None, check_every=10, patience=4):
    """Полный BPR-batch на эпоху; validation используется только для ранней остановки."""
    if epochs < 1 or check_every < 1 or patience < 1:
        raise ValueError('epochs, check_every и patience должны быть положительными')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    user_ids = sorted(train.user_id.unique())
    item_ids = sorted(items.item_id)
    user_index = {u: n for n, u in enumerate(user_ids)}
    item_index = {i: n for n, i in enumerate(item_ids)}
    positive = train[train.rating >= 4]
    if positive.empty:
        raise ValueError('Нет положительных train-оценок')
    rows = positive.user_id.map(user_index).to_numpy()
    cols = positive.item_id.map(item_index).to_numpy()
    seen_mask = np.zeros((len(user_ids), len(item_ids)), dtype=bool)
    seen_mask[train.user_id.map(user_index), train.item_id.map(item_index)] = True
    graph = normalized_graph(rows, cols, len(user_ids), len(item_ids))
    model = LightGCN(len(user_ids), len(item_ids), graph, dimensions, layers)
    # Изолированные узлы не обучаются и не должны выдавать случайный embedding.
    active = torch.sparse.sum(graph, dim=1).to_dense() > 0
    with torch.no_grad():
        model.embedding.weight[~active] = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    users, positives = torch.tensor(rows), torch.tensor(cols)
    history = []
    best_score, best_epoch, best_weights, stale = -1., epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        negatives = torch.tensor(negative_samples(rows, seen_mask, rng))
        optimizer.zero_grad()
        loss = model.bpr_loss(users, positives, negatives, regularization)
        loss.backward()
        optimizer.step()
        # Узел без рёбер мог попасть в negative sampling. Он остаётся холодным.
        with torch.no_grad():
            model.embedding.weight[~active] = 0
        record = {'epoch': epoch, 'loss': float(loss.detach())}
        if validation is not None and (epoch % check_every == 0 or epoch == epochs):
            from .evaluate import ranking_metrics
            model.eval()
            recommender = LightGCNRecommender.from_model(model, train, items)
            relevant = {int(u): set(g.item_id) for u, g in validation[validation.rating >= 4].groupby('user_id')}
            if not relevant:
                raise ValueError('В validation нет положительных оценок')
            score = float(np.mean([ranking_metrics(recommender.recommend(u), ids)['ndcg_at_10']
                                   for u, ids in relevant.items()]))
            record['validation_ndcg_at_10'] = score
            print(f'layers={layers} epoch={epoch} loss={record["loss"]:.4f} val_ndcg={score:.4f}', flush=True)
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= patience:
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    recommender = LightGCNRecommender.from_model(model, train, items)
    return model, recommender, history, best_epoch, best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--user', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    model = LightGCNRecommender.load(args.checkpoint)
    print(json.dumps(model.titles(model.recommend(args.user, args.k)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
