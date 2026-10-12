# Часть 2 — API классификатора скриншотов

FastAPI принимает изображение и вызывает CLIP из `image/clip_classifier.py`.

## Запуск из корня репозитория

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: <http://127.0.0.1:8000/docs>. Первый запрос скачивает веса CLIP (~600 МБ),
поэтому выполняется дольше остальных. Веса хранятся в стандартном кеше Hugging Face;
для отдельного кеша можно задать `HF_HOME="$PWD/.cache/huggingface"` перед запуском.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model
curl -F 'file=@screenshot.png' http://127.0.0.1:8000/predict
```

`POST /predict` возвращает `label`, `score`, `probabilities` и идентификатор модели.
`score` — softmax сходств CLIP; это не калиброванная вероятность правильности.
Фиксированная конфигурация: ViT-B/32, pad, descriptive.

| Код | Значение |
|---|---|
| 200 | Корректное предсказание |
| 400 | Пустой, повреждённый или не являющийся изображением файл |
| 413 | Более 8 МиБ или более 16 млн пикселей |
| 415 | Формат отличается от PNG/JPEG/WebP |
| 422 | Нет обязательного поля `file` |
| 503 | Ошибка доступа к весам или нехватка ресурсов модели |

Ограничение размера проверяется при чтении загруженного файла. Для публичного
развёртывания нужен также лимит всего HTTP-запроса на reverse proxy. Учебный сервер
по умолчанию слушает только localhost. `/health` проверяет процесс, а не готовность весов.
Обработка модели защищена блокировкой: параллельные запросы выполняют инференс последовательно.

## Тестирование

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Тесты API подменяют функцию инференса через `dependency_overrides`, проверяя контракт
и не скачивая веса в CI. Это не проверка качества CLIP. Настоящий инференс проверяется
отдельно скриптом `verification/smoke_api.py`; метрики качества — `image/evaluate.py`.
GitHub Actions запускает тесты на каждый `push` (включая отправленные коммиты),
`pull_request` и ручной запуск. Сам локальный `git commit` без отправки на GitHub
не запускает удалённый workflow.

Источники: [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/),
[GitHub Actions events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).
