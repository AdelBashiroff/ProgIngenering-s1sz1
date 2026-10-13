# Программная инженерия — блок заданий 1

Баширов Адель Тагирович, Казанский федеральный университет, ИТИС, группа 11-411.

| Часть | Реализация | Документация |
|---|---|---|
| 1 — текст | RuBERT, тональность русских отзывов | [text](text/README.md) |
| 1 — аудио | YAMNet, десять классов ESC-10 | [audio](audio/README.md) |
| 1 — изображения | CLIP, тип сайта по скриншоту | [image](image/README.md) |
| 1 — видео | YOLO11, детекция объектов | [video](video/README.md) |
| 1 — локальная LLM | Qwen2.5-0.5B-Instruct | [llm](llm/README.md) |
| 2 — API и тесты | FastAPI + PyTest + GitHub Actions | [api](api/README.md) |
| 3 — RecSys | Content, item-CF, hybrid, popular, Bayesian | [recsys](recsys/README.md) |

## Основное окружение

Python 3.11. Images, API, LLM и RecSys используют общее окружение:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python -m image.evaluate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Отдельно запускаются рекомендации и языковая модель:

```bash
python -m recsys.download_data
python -m recsys.evaluate
python -m recsys.recommender --user 1 --method item_cf
python -m llm.generate 'Зачем нужны тесты API?'
```

У аудио и видео собственные `requirements.txt` и окружения, чтобы не смешивать
зависимости TensorFlow и Ultralytics. Текст работает с корневым окружением.
При первом запуске необходим интернет для загрузки весов; затем можно использовать
кеш и `HF_HUB_OFFLINE=1`. Веса, виртуальные окружения и крупные исходные наборы
в репозиторий не включены. Проверенные скриншоты Images и результаты экспериментов включены.

## Проверки и воспроизводимость

- `tests/`: проверки API, сбора ссылок и алгоритмов/метрик RecSys.
- `.github/workflows/tests.yml`: тесты при push и pull request.
- `verification/api_real_request.json`: настоящий запрос к CLIP через HTTP.
- `verification/environment-macos.txt`: версии проверенного локального окружения.
- `image/results/`, `recsys/results/`, `llm/results/`: результаты реальных запусков.
- `text/results/`, `audio/results/`, `video/results/`: ранее полученные результаты.

Единый отчёт (17 страниц): [PDF](report/Report_Bashirov_11-411.pdf) ·
[DOCX](report/Report_Bashirov_11-411.docx).
Генератор и инструкция сборки находятся в [report/](report/README.md).

## Исходные задания

[Часть 1](https://disk.yandex.ru/i/eMj2rXfaTH0c8Q) ·
[Часть 2](https://disk.yandex.ru/i/xdvx9uKuo9cpQA) ·
[Часть 3](https://disk.yandex.ru/d/k2wM2rv8ueTwFQ)

## Полнота сверки задания

Части 1 и 2 сверены с полными исходными текстами. В части 1 используются два
ML-фреймворка: PyTorch и TensorFlow. RecSys оставлен в текущем объёме по решению
автора проекта; требования за пределами доступной первой страницы не сверялись.
Подробности: [verification/requirements-audit.md](verification/requirements-audit.md).
