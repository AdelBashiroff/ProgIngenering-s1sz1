"""Закрытый набор инструментов. LLM не получает доступ к SQL, файлам или shell."""
import hashlib
from pathlib import Path

from recsys.recommender import DATA, GENRES, Recommender, load_data, temporal_split


class MovieTools:
    def __init__(self, data_dir=DATA):
        ratings, items = load_data(data_dir)
        self.train, _ = temporal_split(ratings)
        self.model = Recommender(self.train, items)
        self.catalog = items.set_index('item_id')
        digest = hashlib.sha256()
        for name in ('u.item', 'u.data'):
            digest.update((Path(data_dir) / name).read_bytes())
        self.checkpoint = Path(__file__).resolve().parents[1] / 'recsys/results/advanced/lightgcn.pth'
        if self.checkpoint.exists():
            digest.update(self.checkpoint.read_bytes())
        self.version = digest.hexdigest()
        self.gnn = None

    def _recommend(self, user_id, method, count):
        if method == 'lightgcn':
            if self.gnn is None:
                from recsys.lightgcn import LightGCNRecommender
                self.gnn = LightGCNRecommender.load(self.checkpoint)
            return self.gnn.recommend(user_id, k=count)
        return self.model.recommend(user_id, method, count)

    def call(self, plan, user_id, profile):
        plan.check_genres()
        if plan.tool == 'refuse':
            return {'items': [], 'note': 'Запрос вне задач каталога MovieLens 100K'}
        avoid = set(profile['avoid']) | set(plan.avoid)
        genres = set(plan.genres or profile['genres']) - avoid
        if plan.genres and not genres:
            return {'items': [], 'tool': plan.tool,
                    'note': 'Запрошенные жанры запрещены текущими ограничениями'}
        row_index = self.model.user_index.get(user_id)
        has_likes = row_index is not None and self.model.likes[row_index].nnz > 0
        personal = plan.tool == 'compare' or (plan.tool == 'recommend' and plan.method != 'popular')
        fallback = personal and not has_likes
        actual_method = 'popular' if fallback else plan.method
        if plan.tool == 'history':
            actual_method = 'history'
        elif plan.tool == 'search':
            actual_method = 'catalog_search'
        elif plan.tool == 'compare' and not fallback:
            actual_method = 'item_cf + lightgcn (rank sum)'
        sources = {}
        if plan.tool == 'history':
            rows = self.train[self.train.user_id == user_id].sort_values(
                ['timestamp', 'item_id'], ascending=[False, True])
            ids = rows.item_id.tolist()
        elif plan.tool == 'search':
            query = plan.query.casefold().strip()
            ids = [int(i) for i, r in self.catalog.iterrows()
                   if not query or query in r.title.casefold()]
        elif plan.tool == 'compare':
            first = self._recommend(user_id, 'item_cf', len(self.catalog))
            second = self._recommend(user_id, 'lightgcn', len(self.catalog))
            # Сначала фильмы, высоко стоящие в обоих списках.
            ranks = {i: n for n, i in enumerate(first)}
            ranks2 = {i: n for n, i in enumerate(second)}
            ids = sorted(set(first + second), key=lambda i: (ranks.get(i, 9999) + ranks2.get(i, 9999), i))
            sources = {i: {'item_cf_rank': ranks.get(i, 9999) + 1,
                           'lightgcn_rank': ranks2.get(i, 9999) + 1} for i in ids}
        else:
            ids = self._recommend(user_id, plan.method, len(self.catalog))
        records = []
        for item_id in ids:
            row = self.catalog.loc[item_id]
            item_genres = [g for g in GENRES if row[g]]
            if avoid.intersection(item_genres) or (genres and not genres.intersection(item_genres)):
                continue
            records.append({'item_id': int(item_id), 'title': row.title,
                            'genres': item_genres, **sources.get(item_id, {})})
            if len(records) >= min(12, max(6, plan.k * 2)):
                break
        return {'items': records, 'tool': plan.tool, 'method': actual_method,
                'requested_method': plan.method, 'fallback_to_popular': fallback,
                'user_known': user_id in self.model.seen,
                'catalog_version': self.version, 'genres': sorted(genres), 'avoid': sorted(avoid),
                'note': 'Каталог 1990-х, не сведения об актуальных кинотеатрах. Жанры из u.item.'}
