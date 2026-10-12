"""Воспроизводимый offline-эксперимент на полном каталоге: python -m recsys.evaluate."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .recommender import load_data, temporal_split, Recommender, METHODS

BASE = Path(__file__).resolve().parent


def ranking_metrics(recommended, relevant, k=10):
    hits = np.array([i in relevant for i in recommended[:k]], dtype=float)
    nhits = int(hits.sum())
    dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
    ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
    ap = np.sum(np.cumsum(hits) / (np.arange(len(hits)) + 1) * hits) / min(k, len(relevant)) if relevant else 0
    return {'precision_at_10': nhits / k, 'recall_at_10': nhits / len(relevant) if relevant else 0,
            'ndcg_at_10': dcg / ideal if ideal else 0, 'map_at_10': ap,
            'hit_rate_at_10': float(nhits > 0)}


def main():
    out = BASE / 'results';out.mkdir(exist_ok=True)
    ratings, items = load_data()
    train, test = temporal_split(ratings)
    start = time.perf_counter(); model = Recommender(train, items)
    fit_s = time.perf_counter() - start
    relevant = {int(u): set(g.item_id) for u, g in test[test.rating >= 4].groupby('user_id')}
    known = set(train.item_id)
    warm = {u: ids & known for u, ids in relevant.items() if ids & known}
    metadata = {'ratings': len(ratings), 'users': ratings.user_id.nunique(), 'items': len(items),
                'train_ratings': len(train), 'test_ratings': len(test), 'eligible_users': len(relevant),
                'warm_eligible_users': len(warm), 'users_without_test_positives': ratings.user_id.nunique() - len(relevant),
                'cold_test_items': len(set(test.item_id) - known),
                'test_positive_pairs_cold_items': int(((test.rating >= 4) & ~test.item_id.isin(known)).sum()),
                'split': 'per-user temporal 80/20, timestamp ties remain together',
                'positive_rating': 4, 'k': 10, 'item_cf_neighbors': 40, 'hybrid_alpha': .5,
                'candidate_set': 'all 1682 catalog items minus all train-seen items',
                'fit_seconds': round(fit_s, 3)}
    rows, details, examples = [], [], {}
    for method in METHODS:
        start = time.perf_counter()
        recs = {u: model.recommend(u, method, 10) for u in relevant}
        elapsed = time.perf_counter() - start
        vectors = [ranking_metrics(recs[u], relevant[u]) for u in relevant]
        frame = pd.DataFrame(vectors)
        result = {'method': method, **frame.mean().to_dict(),
                  'catalog_coverage': len({i for ids in recs.values() for i in ids}) / len(items),
                  'recommendation_ms_per_user': elapsed / len(relevant) * 1000}
        ndcg = frame.ndcg_at_10.to_numpy()
        rng = np.random.default_rng(42)
        bs = ndcg[rng.integers(0, len(ndcg), size=(1000, len(ndcg)))].mean(axis=1)
        result['ndcg_ci95_low'], result['ndcg_ci95_high'] = np.quantile(bs, [.025, .975])
        result['warm_ndcg_at_10'] = np.mean([ranking_metrics(recs[u], warm[u])['ndcg_at_10'] for u in warm])
        rows.append(result)
        for u, values in zip(relevant, vectors):
            details.append({'user_id': u, 'method': method, **values})
        examples[method] = model.titles(model.recommend(1, method))
        print(method, {k: round(v, 4) for k, v in result.items() if k != 'method'}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(details).to_csv(out / 'per_user_metrics.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=int))
    (out / 'recommendations_user1.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    (out / 'cold_start.json').write_text(json.dumps(model.titles(model.recommend(-1)), indent=2))
    # Попарное сравнение с baseline на одних и тех же пользователях.
    pivot = pd.DataFrame(details).pivot(index='user_id', columns='method', values='ndcg_at_10')
    paired = {}
    for method in METHODS[1:]:
        diff = (pivot[method] - pivot['popular']).to_numpy()
        rng = np.random.default_rng(42)
        values = diff[rng.integers(0, len(diff), size=(1000, len(diff)))].mean(axis=1)
        paired[method] = {'ndcg_difference_vs_popular': float(diff.mean()), 'ci95': np.quantile(values, [.025,.975]).tolist()}
    (out / 'paired_comparison.json').write_text(json.dumps(paired, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(table.method, table.ndcg_at_10, color='#356184')
    ax.errorbar(table.method, table.ndcg_at_10,
                yerr=[table.ndcg_at_10-table.ndcg_ci95_low,table.ndcg_ci95_high-table.ndcg_at_10],
                fmt='none', ecolor='black', capsize=4)
    ax.set_ylabel('NDCG@10 (95% bootstrap CI)');ax.set_title('MovieLens 100K — temporal holdout')
    fig.tight_layout();fig.savefig(out / 'ndcg_comparison.png', dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
