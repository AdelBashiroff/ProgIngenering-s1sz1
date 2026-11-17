"""Явные предпочтения и кеш. Текст LLM не становится фактом в памяти."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from recsys.recommender import GENRES


class Memory:
    def __init__(self, path, clock=time.time):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.db = sqlite3.connect(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY, data TEXT NOT NULL,
                revision INTEGER NOT NULL, updated REAL NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY, data TEXT NOT NULL, expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                status TEXT NOT NULL, updated REAL NOT NULL
            );
        ''')

    def profile(self, user_id):
        row = self.db.execute(
            'SELECT data, revision, updated, expires FROM profiles WHERE user_id=?',
            (user_id,)).fetchone()
        if row is None or row[3] <= self.clock():
            return {'genres': [], 'avoid': [], 'revision': 0}
        return {**json.loads(row[0]), 'revision': row[1], 'updated_at': row[2],
                'expires_at': row[3]}

    def update_profile(self, user_id, genres, avoid, ttl_days=30):
        if user_id < 1 or not 1 <= ttl_days <= 365:
            raise ValueError('Неверный пользователь или срок хранения')
        if set(genres) & set(avoid) or not set(genres + avoid).issubset(GENRES):
            raise ValueError('Жанры неизвестны или противоречат друг другу')
        row = self.db.execute('SELECT revision FROM profiles WHERE user_id=?',
                              (user_id,)).fetchone()
        revision = (row[0] if row else 0) + 1
        now = self.clock()
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?)',
                            (user_id, json.dumps({'genres': genres, 'avoid': avoid}),
                             revision, now, now + ttl_days * 86400))
            self.db.execute('DELETE FROM cache')
        return self.profile(user_id)

    def forget(self, user_id):
        with self.db:
            self.db.execute('DELETE FROM profiles WHERE user_id=?', (user_id,))
            self.db.execute('DELETE FROM runs WHERE user_id=?', (user_id,))
            self.db.execute('DELETE FROM cache')

    def cache_key(self, user_id, plan, version):
        content = [user_id, plan, self.profile(user_id), version]
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def cached(self, key):
        row = self.db.execute('SELECT data, expires FROM cache WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row and row[1] > self.clock() else None

    def cache(self, key, value, ttl=3600):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)',
                            (key, json.dumps(value, ensure_ascii=False), self.clock() + ttl))

    def record_run(self, run_id, user_id, status):
        now = self.clock()
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?)',
                            (run_id, user_id, status, now))
            self.db.execute('DELETE FROM cache WHERE expires <= ?', (now,))
            self.db.execute('DELETE FROM runs WHERE updated < ?', (now - 7 * 86400,))
            self.db.execute('DELETE FROM profiles WHERE expires <= ?', (now,))

    def close(self):
        self.db.close()
