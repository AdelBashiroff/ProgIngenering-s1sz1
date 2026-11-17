"""Проверяемые ограничения типовых запросов о фильмах.

Это небольшой domain parser, а не универсальный NLU: сложные отрицания, цитаты
и необычный порядок слов могут распознаться неверно. Права tools он не меняет.
"""
import re

from multi_agent.schemas import Plan


# Метки совпадают с 19 столбцами MovieLens. \w* покрывает русские словоформы.
GENRE_PATTERNS = {
    'unknown': r'unknown|неизвестн\w*\s+жанр\w*',
    'Action': r'action|экш[е]?н\w*|боевик\w*',
    'Adventure': r'adventures?|приключени\w*',
    'Animation': r'animation|анимаци\w*|мульт(?:фильм|ик)\w*',
    "Children's": r"children(?:['’]s)?|детск\w*|для\s+детей",
    'Comedy': r'comedy|комеди\w*',
    'Crime': r'crime|криминал\w*',
    'Documentary': r'documentary|документальн\w*|документалк\w*',
    'Drama': r'drama|драм\w*',
    'Fantasy': r'fantasy|ф[эе]нтези',
    'Film-Noir': r'film[ -]noir|noir|нуар\w*',
    'Horror': r'horror|ужас\w*|хоррор\w*',
    'Musical': r'musical|мюзикл\w*|музыкальн\w*',
    'Mystery': r'mystery|детектив\w*',
    'Romance': r'romance|романти[кч]\w*|мелодрам\w*',
    'Sci-Fi': r'sci[ -]?fi|science\s+fiction|фантастик\w*|фантастическ\w*',
    'Thriller': r'thriller|триллер\w*',
    'War': r'war|военн\w*|войн\w*',
    'Western': r'western|вестерн\w*',
}
NEGATION = re.compile(
    r'\b(?:без|кроме|исключи(?:те|ть)?|не\s*(?:хочу|предлагай(?:те)?|люблю|надо)|'
    r'without|except|exclude|no)\b', re.IGNORECASE)
RESET = re.compile(
    r'\b(?:но|зато|лучше|а|хочу|предпочитаю|предложи|посоветуй|покажи|'
    r'порекомендуй|выбери|but|instead)\b', re.IGNORECASE)
NUMBERS = {'один': 1, 'одна': 1, 'одну': 1, 'одно': 1, 'одного': 1,
           'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5}
COUNT = re.compile(r'\b(?:[1-5]|' + '|'.join(NUMBERS) + r')\b', re.IGNORECASE)


def is_avoided(request, position):
    # Отрицание действует до конца короткого перечисления или нового пожелания.
    prefix = re.split(r'[.;!?\n]', request[:position])[-1][-140:]
    negations = list(NEGATION.finditer(prefix))
    if not negations:
        return False
    tail = prefix[negations[-1].end():]
    return not RESET.search(tail) and len(re.findall(r'\w+', tail)) <= 8


def requested_count(request):
    candidates = []
    for match in COUNT.finditer(request):
        after = request[match.end():]
        # «Сравни две модели, выбери три фильма»: число моделей не есть k.
        if re.match(r'\s*(?:модел|метод|алгоритм|семейств|башн)\w*', after, re.IGNORECASE):
            continue
        token = match.group().casefold()
        candidates.append(int(token) if token.isdigit() else NUMBERS[token])
    return candidates[-1] if candidates else None


def normalize_plan(plan: Plan, request: str) -> Plan:
    """Уточнить только genres/avoid/k; действие, метод и query выбирает LLM."""
    if plan.tool not in ('recommend', 'compare', 'history'):
        return plan
    matches = []
    for genre, pattern in GENRE_PATTERNS.items():
        for match in re.finditer(r'\b(?:' + pattern + r')\b', request, re.IGNORECASE):
            matches.append((match.start(), genre))
    genres, avoid = [], []
    for position, genre in sorted(matches):
        target = avoid if is_avoided(request, position) else genres
        if genre not in target:
            target.append(genre)
    if len(genres) > 4 or len(avoid) > 4:
        raise ValueError('В одном запросе поддерживается не более четырёх жанров и четырёх запретов')
    # Пересечение сохраняется: tools должен вернуть пустой результат при конфликте.
    data = plan.model_dump()
    data.update(genres=genres, avoid=avoid)
    count = requested_count(request)
    if count is not None:
        data['k'] = count
    return Plan.model_validate(data)
