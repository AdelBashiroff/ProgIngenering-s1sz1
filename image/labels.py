"""Классы задачи и нормализация меток из таблицы со ссылками."""

LABELS = ("landing", "shop", "blog")

_ALIASES = {
    "landing": {"landing", "landing page", "landing-page", "lending", "лендинг", "лэндинг", "одностраничник"},
    "shop": {"shop", "store", "online store", "online shop", "e-commerce", "ecommerce", "eshop",
             "магазин", "интернет-магазин", "интернет магазин", "маркетплейс"},
    "blog": {"blog", "блог"},
}


def normalize_label(value):
    """Приводит метку из таблицы к одному из LABELS. Возвращает None, если метка не распознана."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    for label, names in _ALIASES.items():
        if v in names:
            return label
    return None
