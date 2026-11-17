"""Обучение LightGCN с validation и сравнение шести методов на одном test."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .evaluate import ranking_metrics
from .lightgcn import train_model, DEFAULT_CHECKPOINT
from .recommender import DATA, METHODS, Recommender, load_data, temporal_split

OUT = DEFAULT_CHECKPOINT.parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def bootstrap(values, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def evaluate_methods(classic, gnn, test, items):
    relevant = {int(u): set(g.item_id) for u, g in test[test.rating >= 4].groupby('user_id')}
    rows, details = [], []
    for method in (*METHODS, 'lightgcn'):
        latencies, recommendations, vectors = [], {}, []
        for user, truth in relevant.items():
            start = time.perf_counter()
            ids = gnn.recommend(user) if method == 'lightgcn' else classic.recommend(user, method)
            latencies.append((time.perf_counter() - start) * 1000)
            recommendations[user] = ids
            scores = ranking_metrics(ids, truth)
            vectors.append(scores)
            details.append({'user_id': user, 'method': method, **scores})
        frame = pd.DataFrame(vectors)
        ci = bootstrap(frame.ndcg_at_10)
        row = {'method': method, **frame.mean().to_dict(),
               'catalog_coverage': len({i for ids in recommendations.values() for i in ids}) / len(items),
               'recommendation_ms_mean': float(np.mean(latencies)),
               'recommendation_ms_p95': float(np.quantile(latencies, .95)),
               'ndcg_ci95_low': ci[0], 'ndcg_ci95_high': ci[1]}
        rows.append(row)
        print(f'{method}: NDCG@10={row["ndcg_at_10"]:.5f}, Recall@10={row["recall_at_10"]:.5f}', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    ratings, items = load_data()
    train, test = temporal_split(ratings)
    fit, validation = temporal_split(train)
    common = {'dimensions': 32, 'learning_rate': .01, 'regularization': 1e-4, 'seed': 42}
    trials, histories = [], []
    with threadpool_limits(limits=1):
        for layers in (1, 2):
            trial_started = time.perf_counter()
            _, _, history, epoch, score = train_model(fit, items, layers=layers,
                epochs=args.max_epochs, validation=validation, check_every=10, patience=4, **common)
            trials.append({'layers': layers, 'best_epoch': epoch, 'validation_ndcg_at_10': score,
                           'epochs_run': len(history), 'seconds': time.perf_counter() - trial_started})
            histories.extend({'phase': 'validation', 'layers': layers, **r} for r in history)
        selected = max(trials, key=lambda x: x['validation_ndcg_at_10'])
        config = {**common, 'layers': selected['layers'], 'epochs': selected['best_epoch']}
        print(f'Refit: {config}', flush=True)
        train_started = time.perf_counter()
        model, recommender, history, _, _ = train_model(train, items, **config)
        fit_seconds = time.perf_counter() - train_started
        histories.extend({'phase': 'refit', 'layers': config['layers'], **r} for r in history)
        checkpoint = args.out / 'lightgcn.pth'
        recommender.save(checkpoint, model.state_dict(), config)
        classic = Recommender(train, items)
        comparison, per_user = evaluate_methods(classic, recommender, test, items)
    comparison.to_csv(args.out / 'comparison.csv', index=False)
    per_user.to_csv(args.out / 'per_user_metrics.csv', index=False)
    pd.DataFrame(histories).to_csv(args.out / 'training_history.csv', index=False)
    pivot = per_user.pivot(index='user_id', columns='method', values='ndcg_at_10')
    paired = {}
    for baseline in ('popular', 'item_cf'):
        difference = (pivot.lightgcn - pivot[baseline]).to_numpy()
        paired[baseline] = {'ndcg_difference': float(difference.mean()), 'ci95': bootstrap(difference)}
    write_json(args.out / 'paired_comparison.json', paired)
    write_json(args.out / 'selection.json', {'trials': trials, 'selected': config,
        'criterion': 'maximum validation NDCG@10; test unused in selection',
        'check_every': 10, 'patience_checks': 4, 'max_epochs': args.max_epochs})
    relevant = test[test.rating >= 4]
    seen_in_train = set(train[train.rating >= 4].item_id)
    metadata = {'dataset': 'MovieLens 100K', 'ratings': len(ratings), 'users': int(ratings.user_id.nunique()),
        'items': len(items), 'outer_train_ratings': len(train), 'test_ratings': len(test),
        'inner_fit_ratings': len(fit), 'validation_ratings': len(validation),
        'eligible_test_users': int(relevant.user_id.nunique()),
        'eligible_validation_users': int(validation[validation.rating >= 4].user_id.nunique()),
        'positive_train_edges': int((train.rating >= 4).sum()),
        'positive_test_pairs_without_positive_train_item': int((~relevant.item_id.isin(seen_in_train)).sum()),
        'split': 'outer per-user temporal 80/20; inner temporal 80/20 of outer train; ties together',
        'positive_rating': 4, 'k': 10, 'candidate_set': 'all catalog items minus all train-seen items',
        'negative_sampling': 'uniform over all items unseen in the current training partition; one per positive per epoch',
        'optimization': 'Adam, full BPR batch per epoch, no test-based tuning',
        'selected_config': config, 'refit_seconds': fit_seconds,
        'total_seconds': time.perf_counter() - started, 'device': 'cpu', 'torch_threads': 1,
        'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
        'numpy_version': np.__version__, 'platform': platform.platform(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'ratings_sha256': hashlib.sha256((DATA / 'u.data').read_bytes()).hexdigest()}
    write_json(args.out / 'protocol.json', metadata)
    write_json(args.out / 'recommendations_user1.json', recommender.titles(recommender.recommend(1)))
    write_json(args.out / 'cold_start.json', recommender.titles(recommender.recommend(99999)))
    draw_plots(args.out, comparison, pd.DataFrame(histories))
    print(f'Saved: {args.out}; total {metadata["total_seconds"]:.1f}s', flush=True)


def draw_plots(out, table, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(table.method, table.ndcg_at_10,
           color=['#52718b'] * (len(table) - 1) + ['#c87a35'])
    ax.errorbar(table.method, table.ndcg_at_10,
        yerr=[table.ndcg_at_10 - table.ndcg_ci95_low, table.ndcg_ci95_high - table.ndcg_at_10],
        fmt='none', ecolor='black', capsize=4)
    ax.set_ylabel('NDCG@10'); ax.set_title('MovieLens 100K: same temporal test, 95% bootstrap CI')
    fig.tight_layout(); fig.savefig(out / 'ndcg_comparison.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    checks = history[history.phase == 'validation'].dropna(subset=['validation_ndcg_at_10'])
    for layers, group in checks.groupby('layers'):
        ax.plot(group.epoch, group.validation_ndcg_at_10, marker='o', label=f'{layers} graph layers')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation NDCG@10'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'validation_curve.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
