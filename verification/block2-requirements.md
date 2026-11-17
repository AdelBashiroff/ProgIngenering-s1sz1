# Сверка следующего блока

Дата: 17 ноября 2026 года. В этот блок входят мультиагентная система и красное
дополнение RecSys. Предыдущий отчёт первого блока сохранён отдельно.

Полные исходные DOCX получены с публичных ссылок Яндекс Диска, в том числе
дополнительный документ LLM_for Agent. Текст и сведения о цвете сохранены в
assignment-mas.txt, assignment-recsys-upper.txt и assignment-llm-choice.txt.
Красная часть требует современный метод рекомендаций сверх базы; выбрана LightGCN.

| Требование | Реализация и доказательство |
|---|---|
| ТЗ мультиагентной системы | multi_agent/docs/specification.md |
| Единая система, а не отдельные чаты | multi_agent/workflow.py, LangGraph StateGraph, routing и ограниченный цикл исправления |
| Не менее трёх испытанных LLM | multi_agent/results/model_comparison.json, исходные ответы, одинаковые задачи и версии |
| Движок и обоснование выбора | llama.cpp b12359, multi_agent/models.json, runtime/README.md и docs/research.md |
| Маленькая и квантованная модели | Planner Qwen3.5-0.8B Q8, Writer Qwen3.5-2B Q4 |
| Валидатор другого семейства | SmolLM3-3B Q4, проверка конфигурации ModelRouter и отдельный unit test |
| Jev-like Model | Рабочая трактовка LLM-as-a-Judge по уточнению автора проекта; термин в исходнике неоднозначен |
| Фреймворк и варианты архитектуры | LangGraph; сравнение CrewAI, AutoGen и Haystack в docs/research.md |
| Семь C4 и sequence | multi_agent/diagrams, иллюстрации в отдельном отчёте |
| Системные промпты, свои skills и md | multi_agent/prompts, skills, SOUL.md, MEMORY.md |
| Инструменты | MovieTools: recommend/history/search/compare/refuse; shell/SQL/web tools отсутствуют |
| Сохранение явных ограничений | constraints.py уточняет жанры, запреты и число фильмов до tools; сырой LLM benchmark не исправляется |
| 2–3 слоя памяти и актуальность | State в RAM, SQLite TTL-cache, явный профиль с revision/TTL; tests/test_multi_agent.py |
| Варианты памяти и ограничения RAG | docs/research.md, сравнение Graphiti/PageIndex/Mem0/Letta/LangMem |
| Изолированное выполнение | sandbox.py, 6 реальных probes в results/isolation.json, полный прогон системы в sandbox |
| Product, LLM и infrastructure evals | evaluate.py, benchmark_models.py, результаты и определения метрик |
| Метрики, логи, трейсы, алерты, траектории | observability.py, monitor.py, results/monitoring.json и results/observability |
| Непрерывный монитор и HTTP scrape | monitor --interval, настоящий /metrics HTTP 200, несколько последовательных samples |
| Современная RecSys | recsys/lightgcn.py: разреженный граф, propagation, BPR, реальные обученные веса |
| Оценка современной модели | nested temporal validation, refit, единый test для шести методов, paired bootstrap |
| Ограничения RecSys, production, business | recsys/README.md; нет выдуманных CTR/retention или доказанной победы над item-CF |
| Проверки | verification/block2-pytest.txt, block2-independent-review.md, runtime SHA256 verification |
| Репозиторий и отдельный отчёт | ветка block2, report/Block2_Bashirov_11-411.docx и PDF |

## Важные границы проверки

- Корректный JSON достигается в том числе grammar-constrained decoding; это не
  самостоятельное доказательство понимания задачи.
- Метрика writer в модельном benchmark проверяет ID, порядок и непустую причину.
  Точность объяснений не выдаётся за измеренную человеком semantic accuracy.
- Grounding в E2E означает наличие ID в каталоге; текст проверяет несовершенный судья.
- LLM-судья может пропустить неверные ID/повторы. Эти условия проверяет Python-код.
- Временной split задан внутри каждого пользователя, не одной глобальной датой.
- CTR, watch time, retention и production-нагрузка не измерялись на реальных клиентах.
- OTel SDK, JSONL, Prometheus exporter и локальные алерты проверены. Отдельные
  серверы Prometheus/Alertmanager не развёртывались; для них даны конфиги.
- Seatbelt deprecated и не равен отдельному ядру VM. Проверенный launcher привязан
  к macOS; файлы весов и движка для Apple Silicon скачиваются отдельно.

## Реальная трудность запуска

Первый запуск внутри sandbox завершился ошибкой: при gpu_layers=0 llama.cpp всё
ещё пытался создать Metal command queue. Протокол сохранён отдельно в
block2-initial-sandbox-failure.json. Ограничения sandbox не снимались: добавлены
явные --device none, --no-kv-offload, --no-op-offload и --fit off. После этого
изолированный smoke настоящей Qwen выполнился успешно. Первое сравнение моделей
с недоказанным CPU-only режимом сохранено как block2-benchmark-before-cpu-fix.json;
основная таблица строится по повторному прогону с явным CPU-only режимом.

Первый работающий E2E-прогон без словарной проверки дал 4/6 успешных сценариев:
маленькая LLM потеряла запрет жанра и ошиблась на вложенной инструкции. Числа и
ответы сохранены в block2-system-before-guard.json. После этого добавлен
normalize_plan и выполнен повтор того же regression suite. Этот повтор не
считается независимой оценкой на новых пользовательских данных.
