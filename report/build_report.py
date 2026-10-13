"""Создать единый DOCX-отчёт из сохранённых результатов. Требуется python-docx, Pillow."""
import csv
import json
import re
import textwrap
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'

def js(path):return json.loads((ROOT/path).read_text())
def csvrows(path):return list(csv.DictReader((ROOT/path).open()))
def num(v):return f'{float(v):.4f}'.replace('.', ',')
text = js('text/results/metrics_seara__rubert-tiny2-russian-sentiment.json')
text_base = js('text/results/metrics_seara__rubert-base-cased-russian-sentiment.json')
audio = js('audio/results/metrics_mean.json')['closed_set_10_classes']
img = js('image/results/metrics_clip-vit-base-patch32__pad__descriptive.json')
llm = js('llm/results/example.json')
rec = csvrows('recsys/results/comparison.csv')
protocol = js('recsys/results/protocol.json')
api = js('verification/api_real_request.json')
REPO = 'https://github.com/AdelBashiroff/ProgIngenering-s1sz1'
CODE_REF = '204d00aed33d0ff62f1bf3cfc833ec14eb483a9b'
section_number = 0
figure_number = 0

D=Document();s=D.sections[0]
s.page_width=Cm(21);s.page_height=Cm(29.7)
s.top_margin=Cm(1.8);s.bottom_margin=Cm(1.7);s.left_margin=Cm(2.3);s.right_margin=Cm(1.8)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 st=D.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
 st.font.size=Pt(12)
 for e in st.element.xpath('.//w:pBdr'):
  e.getparent().remove(e)
 fonts=st.element.get_or_add_rPr().find(qn('w:rFonts'))
 if fonts is not None:
  for attr in list(fonts.attrib):
   if 'Theme' in attr: del fonts.attrib[attr]
 st.paragraph_format.space_after=Pt(7)
 st.paragraph_format.line_spacing=1.08
D.styles['Heading 1'].font.size=Pt(17);D.styles['Heading 1'].font.bold=True
D.styles['Heading 2'].font.size=Pt(13);D.styles['Heading 2'].font.bold=True
D.styles['Title'].font.size=Pt(23);D.styles['Title'].font.bold=True
D.styles['Caption'].font.size=Pt(10)
s.different_first_page_header_footer=True
s.footer_distance=Cm(0.8)
title_foot=s.first_page_footer.paragraphs[0]
title_foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
title_foot.add_run('Казань 2026').font.size=Pt(12)
foot=s.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
D.core_properties.author='Баширов Адель Тагирович'
D.core_properties.title='Программная инженерия Блок заданий 1'

def p(t,style=None):return D.add_paragraph(t,style)
def h(t):D.add_heading(t,level=2)
def page(t):
 global section_number
 section_number += 1
 heading=D.add_heading(f'{section_number} '+re.sub(r'^\d+\s+', '', t),level=1)
 heading.paragraph_format.page_break_before=True

def link(label, url):
 para = D.add_paragraph()
 anchor = OxmlElement('w:hyperlink')
 anchor.set(qn('r:id'), D.part.relate_to(url, RT.HYPERLINK, is_external=True))
 run = OxmlElement('w:r'); props = OxmlElement('w:rPr')
 color = OxmlElement('w:color');color.set(qn('w:val'),'234A70');props.append(color)
 run.append(props);node=OxmlElement('w:t');node.text=label;run.append(node)
 anchor.append(run);para._p.append(anchor)
 return para

def excerpt(path, first, last, caption):
 """Вставить подлинные строки исходника, убрав только общий отступ."""
 lines=(ROOT/path).read_text().splitlines()
 assert 1 <= first <= last <= len(lines)
 link(f'{caption} — {path}, строки {first}–{last}', f'{REPO}/blob/{CODE_REF}/{path}#L{first}-L{last}')
 code(textwrap.dedent('\n'.join(lines[first-1:last])), size=8.5)
def table(headers, rows, widths=None):
 t=D.add_table(rows=1,cols=len(headers));t.autofit=False
 if widths:
  for col,width in zip(t.columns,widths): col.width=Cm(width)
 for i,v in enumerate(headers):t.rows[0].cells[i].text=str(v)
 for row in rows:
  cells=t.add_row().cells
  for i,v in enumerate(row):cells[i].text=str(v)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr()
  no=OxmlElement('w:cantSplit');trpr.append(no)
  if ri==0:
   repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for ci,c in enumerate(row.cells):
   if widths:c.width=Cm(widths[ci])
   tcpr=c._tc.get_or_add_tcPr()
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
   tcpr.append(borders)
   margins=OxmlElement('w:tcMar')
   for side in ['top','left','bottom','right']:
    e=OxmlElement('w:'+side);e.set(qn('w:w'),'75');e.set(qn('w:type'),'dxa');margins.append(e)
   tcpr.append(margins)
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF2');tcpr.append(shade)
   for para in c.paragraphs:
    para.paragraph_format.space_after=Pt(2);para.paragraph_format.line_spacing=1.0
    if ci > 0 and len(str(c.text)) < 18: para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:run.font.size=Pt(10.5);run.font.bold=(ri==0)
 p('')
 return t

def pic(path,caption,width=14):
 global figure_number
 figure_number += 1
 caption = re.sub(r'^Рисунок \d+ — ', '', caption)
 para=D.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER
 para.paragraph_format.keep_with_next=True
 para.add_run().add_picture(str(ROOT/path),width=Cm(width))
 c=p(f'Рисунок {figure_number} — {caption}','Caption');c.alignment=WD_ALIGN_PARAGRAPH.CENTER

def code(lines, size=9):
 for line in lines.split('\n'):
  para=p(line);para.paragraph_format.space_after=Pt(3)
  for r in para.runs:r.font.name='Courier New';r.font.size=Pt(size)

# 1
for t in ['Казанский федеральный университет','Институт информационных технологий','и интеллектуальных систем']:
 x=p(t);x.alignment=WD_ALIGN_PARAGRAPH.CENTER
p('');p('');p('')
x=p('Программная инженерия','Title');x.alignment=WD_ALIGN_PARAGRAPH.CENTER
x=p('Отчёт по блоку заданий 1','Subtitle');x.alignment=WD_ALIGN_PARAGRAPH.CENTER
x=p('Развёртывание моделей машинного обучения\nAPI и автоматическое тестирование\nРекомендательные системы');x.alignment=WD_ALIGN_PARAGRAPH.CENTER
p('');p('');p('')
x=p('Выполнил\nБаширов Адель Тагирович\nГруппа 11-411');x.alignment=WD_ALIGN_PARAGRAPH.RIGHT

#2
page('1 Цель и состав работы')
p('В этой работе я использовал готовые модели для четырёх задач: анализа отзывов, распознавания звуков, классификации скриншотов и поиска объектов на видео. Затем добавил API для классификатора изображений и сравнил несколько способов рекомендовать фильмы. Код, инструкции запуска и результаты находятся в одном репозитории.')
p('Классификатор сайтов правильно определил 38 из 45 страниц. API принимает изображение и возвращает результат в JSON. В задаче рекомендаций лучший NDCG@10 получен у item-CF: 0,1651. Кроме основных заданий, в проекте есть пример запуска небольшой языковой модели Qwen на ноутбуке.')
table(['Компонент','Решение','Папка'],[
['Текст','RuBERT — тональность отзывов','text/'],['Аудио','YAMNet — классы звуков ESC-10','audio/'],['Изображения','CLIP — тип веб-страницы','image/'],['Видео','YOLO11 — детекция объектов','video/'],['LLM','Qwen2.5-0.5B-Instruct','llm/'],['API','FastAPI, PyTest, GitHub Actions','api/, tests/'],['Рекомендации','Пять сравниваемых алгоритмов','recsys/']], [3,8.5,5.4])
h('Данные и окружение')
p('Текст, аудио и видео были готовы раньше. В отчёт включены их сохранённые результаты; при итоговой проверке для каждой модели выполнено отдельное предсказание. Полные оценки повторно не считались. Скриншоты, эксперименты Images и RecSys, проверка API и запуск Qwen относятся к 12 октября 2026 года.')
p('В работе используются два ML-фреймворка: PyTorch для текста и изображений, TensorFlow для звука. Transformers упрощает загрузку моделей и подготовку входа, TensorFlow Hub даёт готовую YAMNet, а Ultralytics — средства для детекции и разметки кадров. Основное окружение: Python 3.11, macOS, Apple MPS. Аудио и видео запускаются в отдельных окружениях; версии сохранены в verification/environment-macos.txt.')
h('Репозиторий')
p('https://github.com/AdelBashiroff/ProgIngenering-s1sz1')
p('В README каждого модуля описаны ТЗ, установка и запуск. Таблицы предсказаний и метрики сохранены в папках results. Большие веса моделей и исходные наборы данных скачиваются отдельно.')


page('Архитектура проекта')
p('Проект состоит из отдельных Python-модулей. Каждый получает свой тип данных и вызывает готовую модель. Результаты оценки сохраняются в CSV и JSON, графики — в PNG. Базы данных и общего веб-интерфейса здесь нет. Через HTTP доступен классификатор скриншотов, остальные модули запускаются из командной строки.')
pic('report/figures/architecture_modules.png','Модули проекта и их входы и выходы',16.5)
p('API использует ту же функцию predict, что и локальный классификатор. После проверки файла изображение переводится в RGB и передаётся CLIP. Веса и текстовые векторы классов кешируются. Поэтому модель не нужно заново загружать для каждого запроса.')
pic('report/figures/architecture_api.png','Обработка запроса к классификатору',16.5)
p('Схема показывает основной успешный путь и ошибки внутри обработчика. Если поле file не передано, FastAPI возвращает 422 до вызова обработчика. GET /health проверяет только процесс сервера; для проверки весов нужен запрос к /predict.')

page('Стек и организация запуска')
p('Для всех компонентов используется Python 3.11. Основные эксперименты выполнены на macOS; PyTorch использует Apple MPS, если он доступен. Таблица показывает версии проверенного окружения, а не минимально допустимые версии из requirements.txt.')
table(['Область','Технологии и версии','Назначение'],[
['Текст, Images, LLM','PyTorch 2.14.1; Transformers 4.57.6','Готовые веса, токенизация, обработка изображений, расчёт предсказаний'],
['Аудио','TensorFlow 2.21.0; TensorFlow Hub 0.16.1','Загрузка YAMNet и классификация звуков'],
['Видео','Ultralytics 8.4.171; OpenCV 5.0.0.93','YOLO11, чтение и сохранение кадров'],
['HTTP API','FastAPI 0.143.0; Uvicorn 0.54.0; Pydantic 2.14.1','Маршруты сервера, обработка запросов, схема ответа'],
['Работа с данными','NumPy 2.4.6; pandas 2.3.3; SciPy 1.17.1','Таблицы, векторы и разреженные матрицы'],
['Оценка и RecSys','scikit-learn 1.9.1; Matplotlib 3.11.2','Метрики, TF-IDF, косинусное сходство и графики'],
['Изображения и сбор','Pillow 12.3.0; Playwright 1.63.0','Проверка файлов и получение скриншотов в Chromium'],
['Тесты','PyTest 9.1.1; HTTPX 0.28.1','Проверка контракта API и расчётов без скачивания весов'],
['Контроль версий и CI','Git; GitHub Actions; Python 3.11 в CI','Хранение кода и автоматический запуск тестов'],
],[3.1,6.1,7.7])
p('У аудио и видео отдельные виртуальные окружения: зависимости TensorFlow и Ultralytics устанавливаются из README соответствующего модуля. Для Images, API, RecSys и LLM достаточно корневого requirements-dev.txt. В CI устанавливаются только пакеты, нужные тестам; нейросети там не запускаются.')
h('Где находится реализация')
link('Репозиторий проекта', REPO)
link('Зависимости основного окружения', REPO+'/blob/'+CODE_REF+'/requirements.txt')
link('Автотесты', REPO+'/tree/'+CODE_REF+'/tests')
link('Сценарий GitHub Actions', REPO+'/blob/'+CODE_REF+'/.github/workflows/tests.yml')
link('Результаты реальных запусков и проверок', REPO+'/tree/'+CODE_REF+'/verification')

#3
page('2 Анализ тональности текста')
p('Для текста я выбрал анализ русскоязычных отзывов. На вход подаётся строка, на выходе нужны positive, negative или neutral и оценка score. Используется seara/rubert-tiny2-russian-sentiment. Токенизатор разбивает текст на части, BERT учитывает их контекст, а классификатор выбирает одну из трёх меток. Вход ограничен 256 токенами, дообучения нет [3].')
p('В собственном CSV-наборе 151 отзыв, две колонки: text и label. Я сравнил RuBERT-tiny2 и RuBERT-base через один интерфейс Transformers. Accuracy показывает долю верных ответов, macro-F1 одинаково учитывает все три класса. Это удобно, когда число отзывов разных классов различается.')
table(['Модель','Accuracy','Macro-F1'],[['RuBERT-tiny2',num(text['accuracy']),num(text['macro_f1'])],['RuBERT-base',num(text_base['accuracy']),num(text_base['macro_f1'])]],[8.9,4,4])
p('У tiny2 macro-F1 равен 0,7291, что выше выбранного порога 0,70. Но 95% доверительный интервал [0,6566; 0,8023] пересекает этот порог. Интервал разности с base включает ноль, поэтому уверенно объявить tiny2 точнее нельзя. Для проекта оставлена компактная tiny2: в сохранённом запуске она обрабатывала 222 текста/с против 169 у base.')
pic('text/results/confusion_matrix_seara__rubert-tiny2-russian-sentiment.png','Рисунок 1 — Матрица ошибок RuBERT-tiny2 на 151 отзыве',9)
p('Чаще возникают ошибки на нейтральных и смешанных отзывах: например, когда в одной фразе есть и похвала, и жалоба. Набор небольшой и собран вручную, поэтому эти цифры нельзя переносить на любые отзывы.')
p('При повторной проверке фраза «Отличный сервис, всё понравилось!» получила positive со score 0,9964. Этот пример подтверждает, что модель запускается; её качество оценивалось на всём наборе.')
code('python text/sentiment.py "Отличный сервис, всё понравилось!"')

#4
page('3 Классификация звуков')
p('Для звука выбрана YAMNet из TensorFlow Hub: у неё уже есть веса и 521 класс AudioSet. Модель преобразует звук в спектрограмму и обрабатывает её свёрточной сетью MobileNet [4]. Hub удобен тем, что модель загружается вместе с таблицей классов. В этой работе WAV приводится к моно, float32 и 16 кГц, а оценки коротких фрагментов объединяются средним или максимумом.')
p('Для проверки взят ESC-10: 400 WAV-записей по 5 секунд, по 40 примеров на каждый из десяти классов. В таблице метаданных хранятся имена файлов и метки. Названия ESC-10 сопоставлены с AudioSet вручную. В закрытой задаче ответ выбирается только из этих десяти классов. Ни mean, ни max не требуют обучения модели.')
a_max=js('audio/results/metrics_max.json')['closed_set_10_classes']
table(['Агрегация','Accuracy','Macro-F1'],[['mean',num(audio['accuracy']),num(audio['macro_f1'])],['max',num(a_max['accuracy']),num(a_max['macro_f1'])]],[8.9,4,4])
p('С усреднением mean получен macro-F1 0,9426, 95% доверительный интервал [0,9187; 0,9636]. Для accuracy интервал равен [0,9200; 0,9650]. Если разрешить модели выбирать из всех 521 классов, доля совпадений с сопоставленными метками падает до 0,2375. Получается, высокая точность здесь сильно зависит от ограничения списка ответов.')
pic('audio/results/confusion_matrix_mean.png','Рисунок 2 — Матрица ошибок YAMNet с агрегацией mean',10)
p('По матрице ошибок видно, что модель иногда путает огонь, дождь и волны. Возможно, причина в похожем шуме. При повторной проверке файл 1-47273-A-21.wav получил метку sneezing со score 0,2812. Аудио загрузилось и было обработано без ошибки.')

#5
page('4 Детекция объектов на видео')
p('Для видео я использовал YOLO11n и YOLO11s. Детектор обрабатывает каждый кадр и за один проход предсказывает рамки и классы объектов [5]. Ultralytics удобен готовыми весами и средствами запуска и оценки модели. Программа сохраняет видео с рамками, CSV с детекциями и JSON со статистикой. Слежение за отдельным объектом между кадрами не реализовано.')
p('Качество проверено на 500 случайно выбранных изображениях COCO val2017, seed=42. COCO содержит изображения и JSON-разметку с классами и рамками объектов. Метрика AP@[0,5:0,95] учитывает разные пороги совпадения рамок IoU. Разметки для самого видеоролика нет, поэтому AP рассчитан только на изображениях.')
table(['Модель','AP','AP50','AP small','мс/изобр.'],[['YOLO11n','0,415','0,571','0,226','35,1'],['YOLO11s','0,484','0,653','0,310','47,2']],[4.9,3,3,3,3])
p('У YOLO11s AP выше на 0,069, особенно заметна разница для мелких объектов. При этом она работает медленнее. Для видео оставлена YOLO11n. В сохранённом эксперименте выполнены оба критерия ТЗ: AP не ниже 0,30 и скорость не ниже 5 кадров/с.')
table(['Характеристика видео','Результат'],[['Разрешение и частота','3840×2160, 59,91 кадра/с'],['Длительность и объём','29,48 с; 1766 кадров'],['Скорость полной обработки','13,0 кадров/с'],['Инференс модели','21,5 мс/кадр'],['Всего покадровых детекций','48 207'],['Люди / автомобили','24 204 / 23 576 детекций']],[8.5,8.4])
p('Полная обработка идёт со скоростью 13 кадров/с при частоте исходного ролика около 60 кадров/с, то есть не в реальном времени. Время тратится также на чтение, отрисовку и запись 4K-кадров. 48 207 детекций не означают столько же разных объектов: одна машина может попасть в сотни кадров.')
p('Сложнее всего обнаруживать мелкие и частично закрытые объекты в плотных группах. Можно попробовать большее входное разрешение или более крупную модель, но тогда обработка станет медленнее. Отдельных экспериментов с этими изменениями не было.')
p('Для повторной проверки использовано изображение COCO 000000047571.jpg. Найдены три объекта, в том числе horse с confidence 0,9127 и person с confidence 0,8315. Ссылка на исходный ролик находится в video/videos/my_video.txt. Сам ролик заново целиком не обрабатывался.')

#6
page('5 Классификация скриншотов сайтов')
p('Для Images я выбрал определение типа сайта по скриншоту: landing, shop или blog. CLIP ViT-B/32 сравнивает векторы изображения и текстовых описаний классов [2]. Это позволяет задать свои классы без дообучения. Transformers загружает модель и её обработчик входа, а PyTorch выполняет расчёт. В API используется заранее выбранный вариант pad + descriptive.')
p('Из 73 адресов удалось получить 56 скриншотов, ещё 17 страниц не загрузились. При просмотре исключены 11 кадров с заглушками и окнами, закрывающими содержимое. Осталось 45 изображений: 18 лендингов, 9 магазинов и 18 блогов. Отбор сделан до оценки модели, по качеству снимка.')
p('Скриншоты сняты в Chromium при размере окна 1440×900 и сохранены в JPEG. В manifest.csv записаны адрес, метка, путь к файлу, время, HTTP-статус, SHA256 и результат проверки. Сравниваются обрезка crop и дополнение полями pad, а также три набора описаний классов. Все принятые снимки есть в GitHub.')
table(['Обработка','Промпты','Accuracy','Macro-F1'],[[r['fit'],r['prompts'],num(r['accuracy']),num(r['macro_f1'])] for r in csvrows('image/results/comparison.csv')],[3.9,5,4,4])
p('Основной вариант распознал 38 из 45 страниц: accuracy 0,8444, macro-F1 0,8436. Порог ТЗ 0,70 выполнен; 95% доверительный интервал F1 [0,7159; 0,9363] рассчитан по 1000 bootstrap-повторам. Лучший результат у pad + ensemble — 0,8667. Для подтверждения преимущества нужен новый набор: сейчас варианты сравнивались на тех же данных, где оценивались.')
table(['Класс','Precision','Recall','F1','N'],[[label,num(img['per_class'][label]['precision']),num(img['per_class'][label]['recall']),num(img['per_class'][label]['f1-score']),int(img['per_class'][label]['support'])] for label in ['landing','shop','blog']],[4.9,3,3,3,3])

#7
page('6 Ошибки и ограничения Images')
pic('image/results/confusion_matrix_clip-vit-base-patch32__pad__descriptive.png','Рисунок 3 — Матрица ошибок основной конфигурации CLIP',10)
p('Пять блогов модель отнесла к лендингам или магазину. На Stack Overflow и CSS-Tricks вверху есть крупные блоки, похожие на рекламу, а у Ed Zitron — платные подписки. SparkFun получил landing, Tilda — shop. По виду страниц эти ошибки можно объяснить, но точно установить, какие элементы повлияли на модель, по одному ответу нельзя.')
p('На Stack Overflow модель ошиблась со score 0,9196. Поэтому даже высокий score нельзя считать гарантией правильного ответа.')
table(['Порог score','Охват','Accuracy'],[['0,0','100,00%','84,44%'],['0,5','93,33%','88,10%'],['0,7','75,56%','97,06%'],['0,9','44,44%','95,00%']],[5.9,5.5,5.5])
p('Если не принимать ответы со score ниже 0,7, точность на оставшихся страницах повышается до 97,06%. Однако примерно четверть страниц тогда остаётся без ответа. В API этот порог не включён: его ещё нужно проверить на других данных.')
p('После проверки в наборе осталось 44 англоязычных сайта и только один русскоязычный. Сравнивать качество по языкам здесь нет смысла. Две страницы принадлежат Notion, а часть магазинов не удалось снять из-за блокировок. Ещё одно ограничение CLIP — уменьшение входа до 224×224: мелкий текст на странице теряется.')


page('Примеры скриншотов и ответов модели')
p('Оба изображения взяты из сохранённого набора, снятого 12 октября. Ниже показаны исходная метка и ответ основной конфигурации pad + descriptive. Примеры подобраны так, чтобы показать и правильный ответ, и ошибку с высоким score.')
pic('image/data/images/shop/2ec30ac0f31ed5df.jpg','Gymshark — метка shop, ответ shop, score 0,9856',12.8)
p('В кадре видны карточки товаров и элементы каталога. Небольшое окно выбора региона закрывает часть правой карточки, но тип страницы остаётся понятен. Этот же файл использован для API. Адрес: gymshark.com/collections/t-shirts-tops/mens.')
pic('image/data/images/blog/be91c7263c3e8e83.jpg','Stack Overflow Blog — метка blog, ответ landing, score 0,9196',12.8)
p('Верх страницы больше похож на промостраницу, чем на ленту статей. Даже при score 0,9196 ответ оказался неверным. Полная таблица, включая все ошибки, находится в image/results; исходные снимки и метки — в image/data.')

#8
page('7 API и автоматическое тестирование')
p('Для доступа к CLIP я добавил FastAPI. POST /predict принимает файл и возвращает label, score, probabilities и имя модели. GET /health проверяет процесс, GET /model показывает настройки, а /docs открывает Swagger UI. Модель загружается при первом запросе. Блокировка не даёт двум запросам одновременно запустить расчёт.')
pic('report/figures/api_swagger.png','Рисунок 4 — Проверка API через Swagger UI',13)
table(['Код ответа','Условие'],[['200','Корректное изображение и успешный инференс'],['400 / 415','Пустой или повреждённый файл / неподдерживаемый формат'],['413','Файл более 8 МиБ или изображение более 16 млн пикселей'],['422 / 503','Нет обязательного файла / модель недоступна']],[4,12.9])
p(f'Реальный HTTP-запрос со скриншотом магазина завершился кодом 200: label=shop, score={num(api["prediction"]["score"])}; время {api["elapsed_s"]:.3f} с с учётом первого обращения. Ответ сохранён в verification/api_real_request.json.')
p('В проекте 29 тестов: они проверяют ответы API, сбор ссылок, разделение данных и расчёты RecSys, исключение просмотренных фильмов и холодный старт. В тестах API используется подмена модели через dependency_overrides, поэтому GitHub Actions не скачивает веса. Отдельно выполнен HTTP-запрос с настоящей CLIP.')
p('GitHub Actions успешно выполнил тесты коммита a354731 (run 38358132244). Проверки запускаются после push, при pull request и вручную. API рассчитан на локальный запуск. При публикации сервера понадобится также ограничить полный размер HTTP-запроса на reverse proxy.')


page('Фрагменты реализации Images и API')
p('Ниже приведены строки из исходников проекта. Ссылки в заголовках ведут на соответствующие строки GitHub. При сборке отчёта код читается из файлов; убирается только общий отступ фрагмента.')
h('Выбор класса по сходству векторов')
excerpt('image/clip_classifier.py',136,143,'Листинг 1')
p('Сначала получаются векторы изображения и описаний классов. classify_features считает softmax по их сходству, затем argmax выбирает метку. В ответ сохраняются score выбранного класса и значения всех трёх классов. Сам по себе softmax не гарантирует правильность ответа.')
h('Проверка размера загрузки')
excerpt('api/main.py',47,54,'Листинг 2')
p('Обработчик читает не больше лимита плюс один байт: этого достаточно, чтобы обнаружить превышение. Пустой файл даёт 400, слишком большой — 413. Дальше Pillow проверяет формат, декодирование и число пикселей. Ограничение всего HTTP-запроса до разбора multipart остаётся задачей серверной конфигурации.')
h('Вызов модели и обработка сбоя')
excerpt('api/main.py',69,75,'Листинг 3')
p('Блокировка не позволяет нескольким запросам одновременно загружать веса и запускать модель. При ошибке загрузки или выполнения возвращается 503, а успешный результат проходит проверку схемы Prediction.')

page('Проверка API на реальном запросе')
p('13 октября через Swagger UI отправлен JPEG из примера Gymshark. Сервер вернул 200, shop и score 0,9856. На снимке видны команда запроса, адрес, тело ответа и заголовки. Это ответ работающего сервера, а не пример, сгенерированный схемой OpenAPI.')
pic('report/figures/api_prediction.jpg','Ответ CLIP на запрос из Swagger UI',14)
h('Автоматическая проверка контракта')
excerpt('tests/test_api.py',34,41,'Листинг 4')
p('Тест проверяет код ответа, метку, score, сумму значений и имя модели. Он запускается для PNG, JPEG, WebP и PNG с прозрачностью. Здесь модель заменяется фиксированным предиктором, поэтому score 0,8 отличается от настоящего ответа выше. Другие тесты проверяют повреждённые файлы, лимиты и недоступность модели.')
link('Успешный запуск 29 тестов в GitHub Actions', REPO+'/actions/runs/38429429913')

#9
page('8 Постановка задачи рекомендаций')
p('Рекомендательная система подбирает пользователю десять фильмов, которых нет в его истории оценок train. MovieLens 100K содержит 100 000 оценок 943 пользователей для 1682 фильмов. Релевантными считаются оценки 4–5. Демографические признаки не используются.')
table(['Подход','Реализация'],[['Popular','Количество положительных оценок в train'],['Bayesian','Средняя оценка со сглаживанием 20 псевдооценками'],['Content','TF-IDF жанров и cosine с профилем понравившихся фильмов'],['Item-CF','Cosine по общим положительным оценкам; 40 соседей фильма'],['Hybrid','Среднее нормированных content и item-CF, вес 0,5']],[4.1,12.8])
p('Два классических подхода и гибрид построены с NumPy, SciPy и scikit-learn. Popular и Bayesian служат простыми эвристическими baseline. Новому пользователю либо пользователю без положительной истории выдаётся popular. Все ранее оценённые фильмы, включая отрицательные оценки, исключаются.')
h('Временное разделение и оценка')
p('Для каждого пользователя последние 20% событий по времени оставлены в test; одинаковые временные отметки не разделяются. Получено 78 595 train-оценок и 21 405 test-оценок. Ranking-метрики усредняются по 915 пользователям с положительными test-оценками; остальные 28 в среднее не входят.')
p('Кандидаты — весь каталог минус история train. Популярность, средние оценки и collaborative similarity используют только train. Жанры полного каталога считаются заранее доступными. 71 фильм в test отсутствует в train; на них приходится 17 положительных test-пар. Дополнительно рассчитан warm NDCG.')
p('Разделение временное внутри каждого пользователя, но не является глобальным временным срезом: train-события одного пользователя могут быть позже test-событий другого. Для симуляции production нужен единый cutoff и учёт даты появления объекта. Параметры заданы до эксперимента и не оптимизировались по test.')
p('Precision@10 — доля релевантных в списке; Recall@10 — доля найденных релевантных; NDCG@10 учитывает порядок с логарифмическим снижением веса; MAP@10 усредняет precision в позициях попаданий. Покрытие — доля каталога, встретившаяся в рекомендациях. Неизвестная оценка означает отсутствие наблюдаемой релевантности, а не доказанный dislike.')

#10
page('9 Результаты рекомендательных алгоритмов')
table(['Метод','P@10','R@10','NDCG@10','Покрытие'],[[r['method'],num(r['precision_at_10']),num(r['recall_at_10']),num(r['ndcg_at_10']),f"{float(r['catalog_coverage'])*100:.2f}%"] for r in rec],[4.5,2.9,2.9,3.5,3.1])
pic('recsys/results/ndcg_comparison.png','Рисунок 5 — NDCG@10 с 95% bootstrap-интервалами по пользователям',14)
p('Item-CF показывает лучший NDCG@10: 0,1651 против 0,1073 у popular, относительное увеличение около 53,8%. Парная разность +0,0578 имеет 95% ДИ [0,0486; 0,0679]. Гибрид увеличивает покрытие каталога до 25,68% против 14,51% у item-CF, но уступает по NDCG. Content имеет широкое покрытие и низкую точность: одних жанров недостаточно для точного ранжирования.')
p('Интервалы получены 1000 bootstrap-повторами по пользователям с seed=42 на фиксированном разделении. Они не учитывают перенос на иной каталог и аудиторию. Победитель выбран на том же test; для окончательного выбора алгоритма нужен отдельный контрольный период.')
h('Ожидаемый эффект для сервиса')
p('Гипотеза — персонализация повысит долю запусков фильмов и удержание. Offline-рост NDCG не доказывает рост CTR или выручки. Для A/B-теста предлагается рандомизация по пользователю: контроль popular против item-CF, основная метрика — доля запусков из блока рекомендаций, дополнительные — время просмотра, возврат через 7 дней и задержка p95. Размер выборки, полезный эффект и срок определяются до запуска.')

#11
page('10 Уязвимости и локальная языковая модель')
h('Ограничения рекомендательной системы')
p('Холодный старт требует fallback по популярности и стартового выбора интересов. Смещение в пользу популярных фильмов уменьшает разнообразие; следует измерять покрытие и качество по редким жанрам. Подставные оценки могут продвинуть объект, поэтому в реальном сервисе нужны лимиты, дедупликация и поиск аномалий. Эти защитные меры и атаки в эксперименте не реализованы.')
p('MovieLens отражает оценки 1997–1998 годов и не представляет современный каталог. История оценок разрежена, а неизвестные объекты могут оказаться интересными. Для эксплуатации нужны свежие логи показов, мониторинг дрейфа и минимизация персональных данных. Плотная матрица сходств требует O(I²) памяти; при большом каталоге нужен разреженный kNN, ANN либо факторизация.')
h('Развёртывание Qwen')
p('Дополнительно в проекте запущена Qwen2.5-0.5B-Instruct с 0,49 млрд параметров и лицензией Apache-2.0. Небольшая модель подходит для ноутбука. Она работает локально через Transformers, с chat template, float32 и do_sample=False. Вход ограничен 2048 токенами, ответ — 512 новыми токенами.')
p('Тестовый запрос: «Объясни в трёх предложениях, зачем нужны автоматические тесты API». Получен следующий неизменённый ответ:')
for line in llm['answer'].split('\n'):p(line)
table(['Параметр запуска','Значение'],[['Устройство',llm['device']],['Новых токенов',llm['generated_tokens']],['Время генерации',f"{llm['generation_s']} с"],['Скорость',f"{llm['tokens_per_second']} токена/с"]],[8.5,8.4])
p('Qwen выдала три пункта, но написала «ошибки или проблем» и неточно связала тесты с повышением производительности. Тесты помогают найти ошибки, а ускорять сервис нужно отдельно. Этот пример показывает, что локальная генерация работает. Одного запроса недостаточно, чтобы оценить качество модели в целом; полученный код автоматически не исполняется.')

#12
page('11 Воспроизводимость и выводы')
h('Порядок запуска')
code('python3.11 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements-dev.txt\npython -m pytest -q\npython -m image.evaluate\nuvicorn api.main:app --host 127.0.0.1 --port 8000\npython -m recsys.download_data\npython -m recsys.evaluate\npython -m recsys.recommender --user 1 --method item_cf\npython -m llm.generate "Зачем нужны тесты API?"')
p('Uvicorn нужно запустить в отдельном терминале. Для аудио и видео команды приведены в их README. При первом запуске скачиваются веса, затем они берутся из кеша. Скриншоты уже сохранены в репозитории. Для MovieLens есть скрипт загрузки, который записывает SHA256 архива.')
h('Выводы')
p('В результате получились работающие примеры для всех четырёх типов данных. В Images удалось собрать свой набор и получить macro-F1 0,8436 при заданном пороге 0,70. Классификатор доступен через FastAPI, а ответы сервера и основные расчёты проверяются тестами. На практике подготовка входных данных и проверка ошибок оказались такими же существенными частями проекта, как запуск готовой модели.')
p('В RecSys item-CF превосходит popular по NDCG на выбранном временном разделении. Гибрид улучшает покрытие ценой части точности. Перенос результатов на реальный сервис требует нового временного периода и A/B-теста. Небольшие собственные наборы, визуальные смещения и ограничения оценочного протокола явно учтены при интерпретации результатов.')
h('Источники и материалы')
for line in [
'1. Задания: disk.yandex.ru/i/eMj2rXfaTH0c8Q; disk.yandex.ru/i/xdvx9uKuo9cpQA; disk.yandex.ru/d/k2wM2rv8ueTwFQ.',
'2. CLIP: https://huggingface.co/openai/clip-vit-base-patch32',
'3. RuBERT: https://huggingface.co/seara/rubert-tiny2-russian-sentiment',
'4. YAMNet: https://www.tensorflow.org/hub/tutorials/yamnet; ESC-50: https://github.com/karoldvl/ESC-50',
'5. YOLO11: https://docs.ultralytics.com/models/yolo11/; COCO: https://cocodataset.org/',
'6. FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/',
'7. GitHub Actions: https://docs.github.com/en/actions',
'8. MovieLens 100K: https://grouplens.org/datasets/movielens/100k/',
'9. Qwen: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
'10. Исходные метрики и предсказания: каталоги results каждого модуля в репозитории.']:
 para=p(line);para.paragraph_format.space_after=Pt(4)
 for r in para.runs:r.font.size=Pt(10)

D.save(OUT/'Report_Bashirov_11-411.docx')
print(OUT/'Report_Bashirov_11-411.docx')
