"""Отдельный отчёт нового блока из реальных JSON/CSV и исходников.

Сборка DOCX выполняется после сохранения результатов экспериментов.
Зависимости генератора: python-docx и Pillow.
"""
import csv
import json
import math
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
FIG = OUT / 'block2_figures'
REPO = 'https://github.com/AdelBashiroff/ProgIngenering-s1sz1'
SOURCE_REF = 'block2'
BLOCK_REPO = f'{REPO}/tree/{SOURCE_REF}'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_csv(name):
    return list(csv.DictReader((ROOT / name).open(encoding='utf-8')))


def number(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def percent(value):
    return number(float(value) * 100, 1) + '%'


def font(size=28, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrapped(draw, value, max_width, selected_font):
    result = []
    for line in value.split('\n'):
        words, current = line.split(), ''
        for word in words:
            trial = (current + ' ' + word).strip()
            if current and draw.textlength(trial, font=selected_font) > max_width:
                result.append(current)
                current = word
            else:
                current = trial
        result.append(current)
    return result


def diagram(name, title, nodes, edges, height=620):
    """Схема из реальных узлов проекта; координаты не зависят от Graphviz."""
    image = Image.new('RGB', (1600, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), title, font=font(31, True), fill='black')
    positions = {key: (x, y, w, h) for key, x, y, w, h, _ in nodes}
    for first, second, label in edges:
        x1, y1, w1, h1 = positions[first]
        x2, y2, w2, h2 = positions[second]
        c1, c2 = (x1 + w1 / 2, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2 / 2)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        if abs(dx) > abs(dy):
            a = (x1 + (w1 if dx > 0 else 0), c1[1])
            b = (x2 + (0 if dx > 0 else w2), c2[1])
        else:
            a = (c1[0], y1 + (h1 if dy > 0 else 0))
            b = (c2[0], y2 + (0 if dy > 0 else h2))
        draw.line([a, b], fill='#607080', width=4)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        arrow = [b, (b[0] - 16 * math.cos(angle - .45), b[1] - 16 * math.sin(angle - .45)),
                 (b[0] - 16 * math.cos(angle + .45), b[1] - 16 * math.sin(angle + .45))]
        draw.polygon(arrow, fill='#607080')
        if label:
            midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            box = draw.textbbox((0, 0), label, font=font(23))
            width = box[2] - box[0]
            lx, ly = midpoint[0] - width / 2, midpoint[1] - 30
            draw.rectangle((lx - 5, ly - 2, lx + width + 5, ly + 28), fill='white')
            draw.text((lx, ly), label, font=font(23), fill='black')
    for key, x, y, w, h, value in nodes:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill='#F1F5F8',
                               outline='#597084', width=3)
        lines = wrapped(draw, value, w - 26, font(27))
        start_y = y + (h - len(lines) * 34) / 2
        for line in lines:
            draw.text((x + w / 2, start_y), line, font=font(27), fill='black', anchor='mt')
            start_y += 34
    target = FIG / name
    image.save(target)
    return target


def build_diagrams():
    FIG.mkdir(parents=True, exist_ok=True)
    diagram('01-landscape.png', 'C4 System landscape', [
        ('student', 30, 100, 250, 120, 'Пользователь\nPerson'),
        ('author', 30, 520, 250, 120, 'Автор проекта\nPerson'),
        ('agent', 430, 100, 350, 120, 'Помощник по фильмам\nSoftware System'),
        ('eval', 430, 520, 350, 120, 'Evals и наблюдение\nSoftware System'),
        ('recsys', 950, 90, 300, 120, 'Рекомендации\nCF и LightGCN'),
        ('inference', 950, 350, 300, 120, 'Локальный inference\nllama.cpp'),
        ('dataset', 1310, 90, 250, 120, 'GroupLens\nВнешний источник'),
        ('weights', 1310, 350, 250, 120, 'Hugging Face\nВнешние веса'),
    ], [('student', 'agent', 'CLI'), ('author', 'eval', 'проверки'),
        ('agent', 'recsys', 'фильмы'), ('agent', 'inference', 'LLM'),
        ('agent', 'eval', 'телеметрия'), ('dataset', 'recsys', ''),
        ('weights', 'inference', '')], height=700)
    diagram('02-context.png', 'C4 System context', [
        ('user', 40, 90, 320, 125, 'Пользователь\nPerson'),
        ('mas', 530, 175, 500, 170, 'Помощник по фильмам\nSoftware System\nПодбор и проверка ответа'),
        ('engine', 1230, 90, 330, 125, 'llama.cpp\nВнешний inference'),
        ('author', 40, 365, 320, 125, 'Автор проекта\nPerson'),
        ('source', 1230, 365, 330, 125, 'MovieLens 100K\nВнешний датасет'),
    ], [('user', 'mas', 'запрос и ответ'), ('mas', 'engine', 'HTTP JSON'),
        ('author', 'mas', 'evals'), ('source', 'mas', 'read only')])
    diagram('03-container.png', 'C4 Containers', [
        ('app', 40, 90, 360, 130, 'Python приложение\nCLI и LangGraph'),
        ('llama', 1020, 90, 520, 130, 'llama.cpp процесс\n127.0.0.1:18081'),
        ('memory', 40, 380, 360, 130, 'SQLite\nПрофиль и кеш'),
        ('data', 600, 380, 360, 130, 'MovieLens и checkpoint\nТолько чтение'),
        ('obs', 1150, 380, 390, 130, 'Runtime каталог\nSpans и trajectories'),
    ], [('app', 'llama', 'schema + prompt'), ('app', 'memory', 'транзакции'),
        ('data', 'app', 'рекомендации'), ('llama', 'obs', 'логи'), ('app', 'obs', 'события')])
    diagram('04-component.png', 'C4 Components', [
        ('planner', 35, 120, 300, 140, 'Planner и guard\nQwen 0.8B\nPlan schema'),
        ('tools', 430, 120, 300, 140, 'MovieTools\nКаталог и RecSys'),
        ('writer', 825, 120, 300, 140, 'Writer\nQwen 2B\nDraft schema'),
        ('validator', 1220, 120, 335, 140, 'Validator\nSmolLM3\nПравила + LaaJ'),
        ('state', 430, 405, 695, 105, 'State: plan, evidence, draft, issues, attempts'),
    ], [('planner', 'tools', 'план'), ('tools', 'writer', 'факты'),
        ('writer', 'validator', 'Draft'), ('state', 'tools', 'общая память')])
    diagram('05-code.png', 'C4 Code', [
        ('app', 530, 75, 520, 105, 'MovieAssistant\nask и граф переходов'),
        ('router', 40, 315, 330, 180, 'ModelRouter\ngenerate\ngenerate_model\nclose'),
        ('tools', 440, 315, 330, 180, 'MovieTools\ncall\n_recommend\nversion'),
        ('memory', 840, 315, 330, 180, 'Memory\nprofile и cache\nupdate_profile\nforget'),
        ('obs', 1240, 315, 330, 180, 'Observability\nspan и event\nfinish\nsave_metrics'),
    ], [('app', 'router', 'роль'), ('app', 'tools', 'Plan'), ('app', 'memory', 'user_id'), ('app', 'obs', 'run_id')])
    diagram('06-dynamic.png', 'C4 Dynamic', [
        ('start', 30, 120, 230, 110, '1 Запрос\nи профиль'),
        ('plan', 350, 120, 230, 110, '2 План\nи tool'),
        ('write', 670, 120, 230, 110, '3 Черновик\nна фактах'),
        ('judge', 990, 120, 230, 110, '4 Проверка\nID и текста'),
        ('finish', 1310, 120, 250, 110, '5 Ответ\nили отказ'),
        ('retry', 670, 405, 550, 110, 'Замечания → writer\nНе более двух черновиков'),
    ], [('start', 'plan', ''), ('plan', 'write', ''), ('write', 'judge', ''),
        ('judge', 'finish', ''), ('judge', 'retry', 'ошибка'), ('retry', 'write', 'повтор')])
    diagram('07-deployment.png', 'C4 Deployment', [
        ('os', 40, 80, 1510, 100, 'macOS ARM64 — Seatbelt sandbox-exec — ограничение файлов и сети'),
        ('python', 60, 290, 350, 130, 'Python LangGraph\nПроцесс помощника'),
        ('server', 600, 290, 370, 130, 'Дочерний llama-server\nОдна LLM на CPU'),
        ('read', 1130, 220, 420, 130, 'Read only\nКод, каталог, GGUF, checkpoint'),
        ('write', 600, 460, 420, 120, 'Writable runtime\nSQLite, JSONL, метрики'),
    ], [('python', 'server', 'только loopback'), ('read', 'server', 'веса'),
        ('python', 'write', 'результаты')])
    sequence_diagram()


def sequence_diagram():
    image = Image.new('RGB', (1600, 920), 'white')
    draw = ImageDraw.Draw(image)
    roles = [('CLI', 80), ('LangGraph', 350), ('Planner', 610), ('Tools', 870),
             ('Writer', 1130), ('Validator', 1430)]
    for title, x in roles:
        draw.rectangle((x - 65, 20, x + 100, 85), fill='#F1F5F8', outline='#597084', width=3)
        draw.text((x + 17, 38), title, font=font(25, True), fill='black', anchor='mt')
        draw.line((x + 17, 85, x + 17, 900), fill='#BAC2CA', width=3)
    events = [(0, 1, 'ask и профиль'), (1, 2, 'запрос и Plan schema'), (2, 1, 'Plan'),
              (1, 3, 'tool и параметры'), (3, 1, 'evidence'), (1, 4, 'evidence и Draft schema'),
              (4, 1, 'Draft'), (1, 5, 'правила и LLM judge'), (5, 1, 'Verdict и issues'),
              (1, 4, 'один повтор при issues'), (4, 1, 'исправленный Draft'),
              (1, 5, 'повторная проверка'), (5, 1, 'Verdict'), (1, 0, 'status и answer')]
    for n, (source, target, label) in enumerate(events):
        y = 130 + n * 55
        a, b = roles[source][1] + 17, roles[target][1] + 17
        draw.line((a, y, b, y), fill='#607080', width=4)
        sign = 1 if b > a else -1
        draw.polygon([(b, y), (b - sign * 14, y - 7), (b - sign * 14, y + 7)], fill='#607080')
        text_x = (a + b) / 2
        bounds = draw.textbbox((0, 0), label, font=font(23))
        tw = bounds[2] - bounds[0]
        draw.rectangle((text_x - tw/2 - 4, y - 30, text_x + tw/2 + 4, y - 4), fill='white')
        draw.text((text_x, y - 29), label, font=font(23), fill='black', anchor='mt')
    image.save(FIG / '08-sequence.png')


class Report:
    def __init__(self):
        self.doc = Document()
        self.section = self.doc.sections[0]
        for attr, value in [('page_width', 21), ('page_height', 29.7), ('top_margin', 1.65),
                            ('bottom_margin', 1.6), ('left_margin', 2.1), ('right_margin', 1.7)]:
            setattr(self.section, attr, Cm(value))
        for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption']:
            style = self.doc.styles[name]
            style.font.name = 'Times New Roman'
            style.font.size = Pt(11.5)
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.line_spacing = 1.05
            for border in style.element.xpath('.//w:pBdr'):
                border.getparent().remove(border)
            for value in style.element.xpath('.//w:rFonts'):
                for key in list(value.attrib):
                    if 'Theme' in key:
                        del value.attrib[key]
        self.doc.styles['Title'].font.size = Pt(22)
        self.doc.styles['Title'].font.bold = True
        self.doc.styles['Subtitle'].font.size = Pt(15)
        self.doc.styles['Heading 1'].font.size = Pt(16)
        self.doc.styles['Heading 1'].font.bold = True
        self.doc.styles['Heading 2'].font.size = Pt(12.5)
        self.doc.styles['Heading 2'].font.bold = True
        self.doc.styles['Caption'].font.size = Pt(9.5)
        self.section.different_first_page_header_footer = True
        self.section.footer_distance = Cm(.65)
        title_footer = self.section.first_page_footer.paragraphs[0]
        title_footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title_footer.add_run('Казань 2026')
        title_run.font.name = 'Times New Roman'
        title_run.font.size = Pt(12)
        footer = self.section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement('w:fldSimple')
        field.set(qn('w:instr'), 'PAGE')
        footer._p.append(field)
        self.doc.core_properties.author = 'Баширов Адель Тагирович'
        self.doc.core_properties.title = 'Мультиагентный помощник по фильмам и графовые рекомендации'
        self.doc.core_properties.subject = 'Мультиагентная система и рекомендательные модели'
        self.doc.core_properties.comments = ''
        self.doc.core_properties.keywords = ''
        self.doc.core_properties.last_modified_by = 'Баширов Адель Тагирович'
        assembled_at = datetime.now(timezone.utc)
        self.doc.core_properties.created = assembled_at
        self.doc.core_properties.modified = assembled_at
        self.figure_number = 0
        self.page_number = 0

    def p(self, text, style=None):
        return self.doc.add_paragraph(text, style)

    def h(self, text):
        return self.doc.add_heading(text, level=2)

    def page(self, title):
        self.page_number += 1
        heading = self.doc.add_heading(f'{self.page_number} {title}', level=1)
        heading.paragraph_format.page_break_before = True

    def code(self, lines, size=8.5):
        for line in lines.splitlines():
            paragraph = self.p(line or ' ')
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.line_spacing = 1
            for run in paragraph.runs:
                run.font.name = 'Courier New'
                run.font.size = Pt(size)

    def excerpt(self, file, start, end, caption):
        lines = (ROOT / file).read_text().splitlines()
        section = '\n'.join(lines[start - 1:end])
        self.link(f'{caption}  {file}  строки {start}–{end}', f'{REPO}/blob/{SOURCE_REF}/{file}#L{start}-L{end}')
        self.code(textwrap.dedent(section))

    def table(self, headers, rows, widths):
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Cm(width)
        for n, value in enumerate(headers):
            table.rows[0].cells[n].text = str(value)
        for row in rows:
            for cell, value in zip(table.add_row().cells, row):
                cell.text = str(value)
        for row_number, row in enumerate(table.rows):
            properties = row._tr.get_or_add_trPr()
            properties.append(OxmlElement('w:cantSplit'))
            if row_number == 0:
                properties.append(OxmlElement('w:tblHeader'))
            for column, cell in enumerate(row.cells):
                cell.width = Cm(widths[column])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell_pr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement('w:tcBorders')
                for side in ['top', 'left', 'bottom', 'right']:
                    border = OxmlElement('w:' + side)
                    border.set(qn('w:val'), 'single'); border.set(qn('w:sz'), '4')
                    border.set(qn('w:color'), 'D9D9D9'); borders.append(border)
                cell_pr.append(borders)
                margins = OxmlElement('w:tcMar')
                for side in ['top', 'left', 'bottom', 'right']:
                    node = OxmlElement('w:' + side)
                    node.set(qn('w:w'), '65'); node.set(qn('w:type'), 'dxa'); margins.append(node)
                cell_pr.append(margins)
                if row_number == 0:
                    shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E7EDF2'); cell_pr.append(shade)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1.5)
                    paragraph.paragraph_format.line_spacing = 1
                    if len(cell.text) < 23:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.size = Pt(9.6)
                        run.bold = row_number == 0
        self.p('')
        return table

    def image(self, path, caption, width=16.4):
        self.figure_number += 1
        paragraph = self.p('')
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.keep_with_next = True
        picture = paragraph.add_run().add_picture(str(path), width=Cm(width))
        picture._inline.docPr.set('descr', caption)
        caption_paragraph = self.p(f'Рисунок {self.figure_number}  {caption}', 'Caption')
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def link(self, label, url):
        paragraph = self.p('')
        hyperlink = OxmlElement('w:hyperlink')
        hyperlink.set(qn('r:id'), self.doc.part.relate_to(url, RT.HYPERLINK, is_external=True))
        run = OxmlElement('w:r')
        properties = OxmlElement('w:rPr')
        color = OxmlElement('w:color'); color.set(qn('w:val'), '244B6B'); properties.append(color)
        run.append(properties)
        text = OxmlElement('w:t'); text.text = label; run.append(text)
        hyperlink.append(run); paragraph._p.append(hyperlink)
        return paragraph


def main():
    required = ['multi_agent/results/model_comparison.json', 'multi_agent/results/system_evaluation.json',
                'multi_agent/results/isolation.json', 'multi_agent/results/monitoring.json',
                'verification/block2-pytest.txt', 'verification/block2-system-before-guard.json']
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise SystemExit('Сначала выполнить реальные проверки: ' + ', '.join(missing))
    config = read_json('multi_agent/models.json')
    benchmark = read_json(required[0])
    system = read_json(required[1])
    isolation = read_json(required[2])
    monitoring = read_json(required[3])
    protocol = read_json('recsys/results/advanced/protocol.json')
    selection = read_json('recsys/results/advanced/selection.json')
    paired = read_json('recsys/results/advanced/paired_comparison.json')
    comparison = read_csv('recsys/results/advanced/comparison.csv')
    summary = system['summary']
    before_guard = read_json(required[5])['summary']
    if not benchmark.get('complete') or benchmark.get('completed_calls') != 30:
        raise SystemExit('Требуется завершённый benchmark трёх моделей с 30 реальными вызовами')
    test_text = (ROOT / required[4]).read_text()
    test_lines = [line for line in test_text.splitlines() if re.search(r'\d+ passed', line)]
    if not test_lines:
        raise SystemExit('Нет успешной сводки pytest в verification/block2-pytest.txt')
    test_summary = test_lines[-1].strip('= ')
    build_diagrams()
    r = Report()

    # 1 Титульная страница
    for line in ['Казанский федеральный университет', 'Институт информационных технологий',
                 'и интеллектуальных систем']:
        paragraph = r.p(line); paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r.p(''); r.p(''); r.p('')
    paragraph = r.p('Мультиагентный помощник по фильмам', 'Title')
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = r.p('Графовые рекомендации LightGCN', 'Subtitle')
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = r.p('Программная инженерия\nОтчёт по следующему блоку заданий')
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r.p(''); r.p(''); r.p('')
    paragraph = r.p('Выполнил\nБаширов Адель Тагирович\nГруппа 11-411')
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r.p(''); r.p('')

    # 2 ТЗ и результат
    r.page('Задача и результат работы')
    r.p('Система помогает выбрать фильмы из MovieLens 100K. Пользователь пишет запрос, '
        'планировщик выбирает инструмент, отдельная модель составляет объяснение, а проверяющий '
        'сверяет его с данными. Рекомендации рассчитываются алгоритмом, названия и жанры берутся '
        'из каталога. LLM не составляет список фильмов по памяти.')
    r.p(f'В повторном разработческом прогоне выполнено {summary["cases"]} сценариев, доля успешных — '
        f'{percent(summary["success_rate"])}. Для рекомендаций обучена LightGCN: NDCG@10 '
        'равен 0,1664 против 0,1651 у item-CF. Уверенного преимущества над item-CF по '
        'парному bootstrap нет, зато покрытие каталога выше. Эти результаты относятся к учебным '
        'данным и локальному запуску.')
    r.h('Техническое задание')
    r.table(['Пункт', 'Требование и критерий'], [
        ['Вход', 'Запрос до 1 500 символов, user_id, явные предпочтения и исключённые жанры'],
        ['Функции', 'Рекомендации, история оценок, поиск по названию и сравнение item-CF с LightGCN'],
        ['Ответ', 'От 1 до 5 фильмов с ID, названием, жанрами и кратким объяснением; без ID вне evidence'],
        ['Архитектура', 'Один граф LangGraph; planner, tools, writer, validator; проверяющий из другого семейства'],
        ['Ограничения', 'Не более двух черновиков; ошибки видны в статусе; нет произвольного shell/SQL'],
        ['Данные', 'MovieLens 100K; train/test по времени; параметры GNN выбираются по validation'],
        ['Проверка', 'Три реальные LLM, E2E-сценарии, тесты, sandbox-проверки и наблюдаемость'],
    ], [3.1, 13.7])
    r.p('Система не ищет сеансы в кинотеатрах и не знает актуальные премьеры: оценки MovieLens '
        'относятся к 1997–1998 годам. Новый блок добавлен к существующему проекту отдельно; '
        'предыдущий отчёт и результаты базовой части не заменяются.')
    r.link('Репозиторий нового блока', BLOCK_REPO)
    r.link('Задание по мультиагентной системе', 'https://disk.yandex.ru/i/PBKPDllZAofp8w')
    r.link('Задание по рекомендательным системам', 'https://disk.yandex.ru/i/Oqaai4SoM5ChzQ')

    # 3 Движок, фреймворк и модели
    r.page('Выбор движка моделей и фреймворка')
    r.p('Для локального инференса выбран llama.cpp. Он запускает GGUF без внешнего API; '
        'можно точно зафиксировать бинарник и файл весов. Ollama проще в установке и управлении '
        'моделями, но добавляет отдельный слой сервиса. vLLM полезен при серверном обслуживании '
        'многих запросов; здесь один пользователь и CPU, поэтому это преимущество не проверялось.')
    r.p(f'Использован llama.cpp {config["engine"]["tag"]}, commit {config["engine"]["commit"]}. '
        f'Контекст — {config["inference"]["context_size"]} токенов, temperature=0, seed=42, '
        f'{config["inference"]["threads"]} CPU-потока. Версии весов, размер и SHA256 записаны '
        'в multi_agent/models.json. В памяти находится одна LLM: при смене роли сервер '
        'перезапускается с нужными весами. Это снижает расход RAM, но увеличивает полное время.')
    model_rows = []
    for key, model in config['models'].items():
        roles = ', '.join(role for role, target in config['roles'].items() if target == key)
        model_rows.append([model['name'], model['quantization'], number(model['size_bytes'] / 1e9, 2) + ' ГБ', roles])
    r.table(['Модель', 'Квантизация', 'Файл', 'Роль'], model_rows, [5.0, 3.0, 2.3, 6.5])
    r.p('Маленькая Qwen разбирает запрос, более крупная формулирует ответ. Валидатор SmolLM '
        'относится к другому семейству; это проверяется при создании ModelRouter. Размер B '
        'означает миллиарды параметров, а Q4/Q8 описывают формат квантизации весов, а не число '
        'параметров. Скорость и корректность проверяются на одинаковых заданиях.')
    r.p('У Qwen 0.8B и 2B одинаковая общая accuracy в этом наборе. Writer 2B выбран '
        'по более полным русским причинам в двух просмотренных примерах; '
        'статистического преимущества такой разбор не доказывает.')
    r.p('При первом sandbox-запуске выяснилось, что n-gpu-layers=0 не отключает '
        'Metal для KV и отдельных операций. Указаны device=none, no-kv-offload, '
        'no-op-offload и fit=off. После этого все 30 модельных вызовов повторены '
        'на CPU. Первичный неудачный запуск и прежний замер сохранены в verification.')
    r.h('Почему LangGraph')
    r.p('Нужны общее состояние, явные переходы и ограниченная доработка после отказа проверяющего. '
        'LangGraph выражает это непосредственно узлами и рёбрами. Рассмотрены CrewAI Flows '
        'для событийных процессов и команд агентов, Haystack для retrieval-pipelines и AutoGen '
        'для общения агентов. Здесь явный небольшой граф проще проследить и тестировать. '
        'Альтернативы сравнивались по документации; отдельный скоростной benchmark фреймворков не проводился.')
    r.p('Полностью свободный чат агентов увеличил бы число вызовов и усложнил завершение. '
        'ReAct удобен, когда шаги заранее неизвестны, но требует дополнительных ограничений. '
        'В выбранном варианте граф задаёт границы, а LLM выбирает действие внутри них.')

    # 4 Изоляция
    r.page('Изоляция и проверка запретов')
    r.p('LLM получает только закрытый список инструментов. Этого недостаточно для изоляции '
        'процесса: ошибка в приложении или библиотеке может обратиться к файлам или сети. '
        'На macOS используется Seatbelt через sandbox-exec. Дочерний llama-server наследует '
        'ограничения. Разрешены чтение кода и весов, запись в runtime и локальное соединение '
        'с inference; внешняя сеть и посторонние файлы ограничены политикой.')
    r.table(['Вариант', 'Граница и цена'], [
        ['Контейнер', 'Изолирует процессы и файловое окружение, но делит ядро хоста; одного Dockerfile недостаточно'],
        ['Runtime sandbox', 'Ограничения ОС без гостевой системы; зависит от политики и конкретной платформы'],
        ['gVisor', 'Linux-интерфейс через userspace kernel; совместимость и overhead, не нативный вариант macOS'],
        ['VM', 'Отдельное гостевое ядро; больше RAM, образ ОС и обслуживание'],
        ['MicroVM / Firecracker', 'Сокращённая VM для недоверенных задач; Firecracker требует Linux/KVM'],
        ['Docker Sandboxes', 'Отдельная sandbox-среда с microVM и сетью; отличается от обычного контейнера'],
        ['E2B / OpenSandbox / Kata', 'Удалённая sandbox или VM-изоляция workloads; дополнительная инфраструктура и управление'],
    ], [4.4, 12.4])
    r.h('Результат фактических проверок')
    checks = isolation.get('checks', [])
    r.table(['Проверка', 'Ожидание', 'Результат'],
            [[c['name'], str(c.get('expected', '')), 'пройдена' if c['passed'] else 'ошибка'] for c in checks],
            [6.6, 7.1, 3.1])
    r.p('Протокол: multi_agent/results/isolation.json. Проверка запрета означает реальную '
        'попытку операции, а не наличие файла политики. Seatbelt — устаревающий механизм '
        'macOS; он выбран для этого локального учебного запуска. Это не VM и не '
        'доказательство готовности к выполнению произвольного недоверенного кода в production.')

    # 5-8 все семь C4 + sequence
    r.page('Окружение и границы системы')
    r.p('Landscape показывает окружение проекта, context — границу системы и внешние роли. '
        'Данные скачиваются при подготовке; обычному запросу не требуется обращаться к '
        'GroupLens или Hugging Face. Общение приложения с локальным inference идёт по HTTP.')
    r.image(FIG / '01-landscape.png', 'Окружение локального помощника')
    r.image(FIG / '02-context.png', 'Пользователь разработчик и граница системы')
    r.p('Источник схем — реальные модули проекта. Редактируемые Mermaid-версии '
        'находятся в multi_agent/diagrams/01-landscape.mmd и 02-context.mmd.')

    r.page('Контейнеры и компоненты')
    r.p('Container в C4 означает выполняемый или хранящий данные компонент и не обязательно '
        'Docker-контейнер. Приложение и llama-server — разные процессы. SQLite хранит профиль '
        'и кеш, MovieLens и checkpoint используются для чтения, runtime — для результатов.')
    r.image(FIG / '03-container.png', 'Процессы и хранилища локального запуска')
    r.image(FIG / '04-component.png', 'Роли одного сценария и общее состояние')
    r.p('Узлы связаны в один workflow. Tools исполняет обычный Python-код; этот узел не является '
        'ещё одним LLM-агентом. После проверки переход выбирается по issues и счётчику attempts.')

    r.page('Код и динамика запроса')
    r.image(FIG / '05-code.png', 'Основные классы и используемые методы')
    r.p('MovieAssistant управляет графом и передаёт состояние. ModelRouter отвечает за выбор '
        'модели и локальный HTTP. MovieTools возвращает факты каталога. Memory и '
        'Observability вынесены отдельно, чтобы их поведение проверялось без LLM.')
    r.image(FIG / '06-dynamic.png', 'Нумерованный сценарий запроса и ограниченный повтор')
    r.p('Пустой результат инструмента завершает запрос со статусом no_results. После '
        'отклонения writer получает замечания и может сделать второй черновик. После '
        'повторной неудачи система возвращает rejected и пустой список, не публикуя '
        'непроверенный ответ. Ошибка движка или схемы отмечается как error.')

    r.page('Развёртывание и последовательность')
    r.image(FIG / '07-deployment.png', 'Проверенный локальный вариант развёртывания')
    r.image(FIG / '08-sequence.png', 'Последовательность вызовов и возврат замечаний')
    r.p('Это семь видов диаграмм C4: landscape, context, container, component, code, dynamic '
        'и deployment, а также отдельная sequence diagram. Код не запускает три модели '
        'параллельно; последовательная смена весов учитывается при оценке полного времени.')

    # 9 prompts tools validation
    r.page('Промпты инструменты и проверка ответа')
    r.p('Системные инструкции лежат в prompts/planner.md, writer.md и validator.md. '
        'Планировщик возвращает Plan: разрешённый tool, метод RecSys, жанры, исключения, '
        'поисковую строку и k. Автор получает только evidence инструмента и замечания '
        'предыдущей проверки. JSON Schema задаётся движку; Pydantic повторно проверяет '
        'тип, диапазоны и лишние поля на стороне приложения. После планировщика '
        'normalize_plan из constraints.py уточняет явно названные жанры, запреты и k '
        'обычным кодом; tool, метод и query по-прежнему выбирает LLM. Raw и '
        'нормализованный план записываются отдельным событием constraint_guard.')
    r.table(['Файлы', 'Содержание'], [
        ['SOUL.md', 'Роль помощника, границы каталога и запрет выдумывать факты'],
        ['MEMORY.md', 'Слои памяти, сроки жизни и правила обновления'],
        ['skills/recommend/SKILL.md', 'Подбор фильмов по evidence с учётом ограничений'],
        ['skills/compare/SKILL.md', 'Сравнение двух алгоритмов без выдуманных численных обещаний'],
        ['skills/catalog/SKILL.md', 'Работа с каталогом, поиском и историей'],
    ], [6.6, 10.2])
    r.p('Инструменты: recommend, history, search, compare и refuse. Названия и параметры '
        'проходят allowlist. Модель не получает shell, произвольный SQL и чтение URL. '
        'Skill — инструкция применения доступного инструмента; сам Markdown не добавляет прав.')
    r.h('Сначала правила затем отдельная модель')
    r.excerpt('multi_agent/schemas.py', 39, 52, 'Листинг 1  Детерминированная проверка ID')
    r.p('Отдельно SmolLM проверяет опору объяснений на evidence. Другой размер Qwen не '
        'выполнил бы требование другого семейства. Судья может ошибаться, поэтому его '
        'одобрение не отменяет проверку ID и схемы.')
    r.p('Термин Jev-like в задании не расшифрован. Рабочая трактовка здесь — LaaJ, '
        'LLM-as-a-Judge: отдельная модель-судья. Это предположение о термине, '
        'которое стоит подтвердить у преподавателя.')

    # 10 memory
    r.page('Память и актуальность данных')
    r.p('Нужны три срока жизни данных. State переносит информацию внутри одного запроса. '
        'Кеш ускоряет повторные инструменты. Профиль сохраняет только явно введённые '
        'предпочтения пользователя; LLM не записывает свои догадки как факты.')
    r.table(['Слой', 'Содержимое', 'Обновление'], [
        ['Состояние run', 'request, plan, evidence, draft, issues, attempts', 'Новое на каждый запуск; траектория сохраняется для разбора'],
        ['Кеш SQLite', 'Ответы инструментов', 'TTL 1 час; ключ включает user_id, план, профиль и версию данных/модели'],
        ['Профиль SQLite', 'Предпочтительные и исключённые жанры', 'По явной команде; revision и timestamp; по умолчанию TTL 30 дней'],
    ], [3.8, 5.7, 7.3])
    r.p('При изменении профиля кеш очищается. Истёкшие значения не возвращаются; '
        'очистка выполняется при записи сводки run. Сводки запусков хранятся семь дней. '
        'Команда forget удаляет профиль и сводки данного пользователя и очищает кеш. '
        'Удаление сырых JSONL-логов требует отдельной политики: это не следует путать '
        'с удалением профиля из SQLite.')
    r.h('Почему SQLite и почему одного RAG недостаточно')
    r.p('Каталог и ограничения уже структурированы. Точный ключ user_id и версия '
        'предпочтения надёжнее поиска похожей фразы: старое и новое ограничение могут '
        'быть семантически близки. RAG полезен для описаний и рецензий, но не решает '
        'сам актуальность, противоречия и удаление памяти. Поэтому он не выбран основным '
        'хранилищем этого прототипа.')
    r.table(['Альтернатива', 'Для чего полезна и почему отложена'], [
        ['Graphiti', 'Темпоральные факты и связи; для небольшого профиля слишком много инфраструктуры'],
        ['PageIndex', 'Дерево длинного документа; текущие источники — таблицы, а не корпус отчётов'],
        ['Mem0 / LangMem', 'Извлечение памяти из диалогов; возможны ошибочные постоянные факты'],
        ['Letta', 'Долгие диалоги и редактируемые блоки; сейчас хватает явного state и SQLite'],
    ], [4.3, 12.5])
    r.p('Недостатки выбора: SQLite ограничивает конкурентную запись; нет распределённой '
        'памяти, автоматического извлечения фактов и полной истории противоречий. '
        'При росте проекта понадобятся серверное хранилище и отдельный поиск по текстам.')

    # 11 observability
    r.page('Наблюдаемость и уровни оценки')
    r.p('Выбран локальный стек OpenTelemetry SDK, метрики в формате Prometheus и JSONL. Логи описывают события, '
        'span — интервал операции, trace связывает операции одного запроса. Траектория '
        'хранит порядок решений, tools, проверки и доработки. Это позволяет понять причину '
        'отказа, которую не видно по одному времени ответа.')
    r.table(['Сигнал', 'Что сохраняется'], [
        ['spans.jsonl', 'OTel spans узлов и инструментов, время, trace_id и run_id'],
        ['trajectories.jsonl', 'План, evidence, черновики, решения валидатора, ошибки и статус'],
        ['metrics.prom', 'Исходы, latency, токены, tools/cache и отказы валидатора'],
        ['alerts.jsonl', 'Локальные события AgentFailed и AgentSlow при выполнении условий'],
    ], [4.7, 12.1])
    r.p('Рассмотрены LangSmith, Langfuse и Phoenix. Они удобнее для визуального анализа '
        'LLM-трасс и экспериментов. Обычный локальный стек не требует облачного аккаунта; '
        'его цена — собственные сводки и менее удобный интерфейс. Алерт в локальный файл '
        'не равен доставке уведомления в мессенджер.')
    r.h('Что оценивается отдельно')
    r.table(['Уровень', 'Метрики'], [
        ['Продукт', 'Выполнение запроса, жанр, количество фильмов, корректный отказ'],
        ['LLM', 'JSON/schema, выбор tool, фактическая опора, LaaJ и согласие с проверяемой разметкой'],
        ['Весь workflow', 'E2E success, число попыток, ошибки, запрещённые действия и память'],
        ['Инфраструктура', 'Полное p50/p95, время LLM и загрузки, токены, RSS, доступность'],
        ['RecSys', 'NDCG, Recall, Precision, MAP, Hit Rate и покрытие каталога'],
        ['Бизнес', 'Будущий CTR, старты и время просмотра, возврат через 7 дней, жалобы'],
    ], [3.8, 13.0])
    r.p('Фактическая проверка наблюдаемости сохранена в multi_agent/results/monitoring.json. '
        'Evals повторяются после изменения модели, промпта, инструмента или каталога. '
        'Бизнес-метрики требуют реальных показов и A/B-теста; локальная accuracy не '
        'доказывает рост просмотров или удержания.')
    snapshot = monitoring['snapshot']
    exporter = monitoring['exporter']
    r.p(f'При контролируемой недоступности движка получены {snapshot["spans"]} spans, '
        f'{snapshot["trajectory_events"]} события траектории и {snapshot["alerts"]} локальный алерт. '
        f'Счётчик ошибки реально прочитан с /metrics, HTTP {exporter["http_status"]}; '
        f'монитор сохранил {exporter["monitor_samples"]} последовательных измерения '
        f'с интервалом {exporter["interval_s"]} с. Этот отдельный fault experiment '
        'не включён в шесть обычных E2E-сценариев.')

    # 12 actual LLM + E2E
    r.page('Результаты моделей и всей системы')
    model_results = benchmark['models']
    n_tasks = model_results[0]['aggregate']['n_cases']
    r.p(f'Три LLM получили одинаковые {n_tasks} задач: 4 planner, 2 writer и 4 judge. '
        'JSON ограничен схемой движка. Writer оценивается по ID, порядку и непустой '
        'причине, а не по смысловой точности объяснения. Сырые ответы, токены и '
        'время сохранены; это небольшой функциональный набор, не общий рейтинг LLM.')
    r.table(['Модель', 'Верно', 'Schema', 'Среднее с', 'p95 с'], [
        [m['name'], percent(m['aggregate']['task_accuracy']), percent(m['aggregate']['schema_accuracy']),
         number(m['aggregate']['mean_inference_s'], 2), number(m['aggregate']['p95_inference_s'], 2)]
        for m in model_results], [5.8, 2.5, 2.5, 3.0, 3.0])
    r.p('Каждый judge решил 2/4 задачи — 50%, ложное одобрение — 2/3 неверных примеров. '
        'Неизвестный ID пропустили все, дубликат — Qwen 2B и SmolLM. Поэтому ID и '
        'повторы обязательно проверяются кодом. SmolLM отклонил выдуманный сюжет; '
        'это один пример смысловой проверки, а не гарантия обнаружения всех галлюцинаций.')
    r.h('End to end')
    r.table(['Показатель', 'Результат'], [
        ['Сценариев', summary['cases']], ['Успешность', percent(summary['success_rate'])],
        ['Выбор инструмента', percent(summary['tool_accuracy'])],
        ['Непустой ответ с ID каталога', percent(summary['grounding_rate'])],
        ['p50 / p95 полного запроса', number(summary['p50_s'], 2) + ' / ' + number(summary['p95_s'], 2) + ' с'],
        ['Ошибок', summary['errors']],
        ['Пиковый RSS процесса Python', number(summary['peak_rss_self_mb'], 1) + ' МиБ; без llama-server'],
        ['Тесты', test_summary],
    ], [7.3, 9.5])
    failures = [case for case in system['cases'] if not case['success']]
    if failures:
        r.p('Неуспешные сценарии: ' + '; '.join(
            str(case['id']) + ' — ' + str(case.get('error') or case.get('result', {}).get('status', 'ошибка'))[:160]
            for case in failures[:3]) + '. Подробные планы и причины сохранены в system_evaluation.json; '
            'неуспешные случаи остаются в знаменателе success rate.')
    else:
        r.p('В проверенном наборе все сценарии выполнили заданные условия. Это не доказывает '
            'безошибочность на произвольных запросах: состав и размер набора ограничены.')
    r.p('E2E success требует ожидаемых статуса и инструмента, запрошенного метода и '
        'числа фильмов, соблюдения жанров и исключений, отсутствия просмотренного и '
        'корректных ID. Grounding — доля всех шести сценариев с непустым ответом '
        'и ID каталога; пустой или rejected даёт ноль. Смысл причин проверяет '
        'модель-судья; это не ручная оценка отсутствия галлюцинаций. '
        'p50/p95 рассчитаны по запускам, вернувшим result JSON.')
    r.p('В injection проверяющий записал положительные комментарии в поле issues. '
        'Система трактует непустое issues как отказ и после двух попыток не выдала '
        'черновик. Это безопасный отказ, но продуктовая задача не выполнена. '
        'Все фактически выданные ID корректны: 83,3% означает пять непустых '
        'сценариев из шести, а не 16,7% выдуманных фильмов.')
    r.p(f'До guard этот же набор дал {percent(before_guard["success_rate"])} '
        f'({round(before_guard["success_rate"] * before_guard["cases"])}/{before_guard["cases"]}), '
        'хотя tool accuracy была 100%: запрос комедий получил фильм Horror, '
        'а сценарий с посторонней инструкцией завершился rejected. После исправлений '
        'повторён тот же разработческий regression suite. Это проверка до и после, '
        'а не независимый held out тест. Новый набор нужен для оценки переноса.')
    r.p('Unit-тесты проверяют контракты и ветки графа с подменой LLM; реальные '
        'вызовы сохранены отдельно в model_comparison.json и system_evaluation.json.')

    # 13 model
    r.page('Современные рекомендации и LightGCN')
    r.p('В базовой части уже были popular, Bayesian, content, item-CF и hybrid. '
        'Для усложнения рассмотрены три направления: Two-Tower, графовые модели и '
        'языковые рекомендательные модели. Two-Tower раздельно кодирует пользователя '
        'и объект и удобен для предрасчёта каталога и ANN. P5 переводит историю и '
        'задачи в text-to-text формат, но требует более дорогого обучения и контроля '
        'генерируемых ID. Эти два подхода здесь изучены, а не обучены.')
    r.p('Реализован LightGCN по статье He et al., SIGIR 2020 [10]. Он подходит к '
        'имеющимся оценкам: пользователь и фильм — разные типы узлов, положительная '
        'оценка 4–5 — ребро. Модель передаёт нормированные векторы по графу и '
        'усредняет представления всех слоёв. Это GNN, хотя в ней нет нелинейных активаций '
        'и матриц преобразования на каждом слое.')
    r.code('A_norm = D^(-1/2) A D^(-1/2)\nE(l+1) = A_norm E(l)\nE_final = mean(E(0), ..., E(L))\nscore(u, i) = dot(E_final[u], E_final[i])', size=10)
    r.p('Обучаются начальные embeddings ID, 32 числа на узел. BPR увеличивает разность '
        'score положительного и случайного неоценённого фильма. Negative sampling исключает '
        'всю историю текущего train, включая низкие оценки. Пропуск не означает dislike; '
        'он используется как учебный контраст. Граф разреженный, исходные веса и итоговые '
        'векторы сохранены в checkpoint.')
    r.h('Разделение без подбора по test')
    r.table(['Этап', 'Данные и назначение'], [
        ['Внешний train / test', f'{protocol["outer_train_ratings"]} / {protocol["test_ratings"]}; последние 20% каждого пользователя — test'],
        ['Внутренний fit / validation', f'{protocol["inner_fit_ratings"]} / {protocol["validation_ratings"]}; ещё один temporal split внешнего train'],
        ['Подбор', '1 или 2 слоя; validation NDCG каждые 10 эпох; максимум 200; patience 4 проверки'],
        ['Refit', 'Выбранная конфигурация обучается заново на всём внешнем train; затем один test'],
    ], [5.0, 11.8])
    r.p('Одинаковые timestamps остаются в одной части. Это split по времени внутри '
        'пользователя, а не глобальная симуляция сервиса: событие train одного человека '
        'может быть позже test другого. Параметры пяти классических методов сохранены '
        'прежними; сравнение выполняется на одном test и полном каталоге.')

    # 14 results
    r.page('Сравнение рекомендательных алгоритмов')
    r.table(['Метод', 'P@10', 'R@10', 'NDCG@10', 'MAP@10', 'Покрытие'], [
        [row['method'], number(row['precision_at_10']), number(row['recall_at_10']),
         number(row['ndcg_at_10']), number(row['map_at_10']), percent(row['catalog_coverage'])]
        for row in comparison], [3.6, 2.5, 2.5, 3.0, 2.5, 2.7])
    r.image(ROOT / 'recsys/results/advanced/ndcg_comparison.png', 'Один temporal test и 95 процентные bootstrap интервалы', 15.5)
    difference = paired['item_cf']
    r.p('Оцениваются 915 пользователей с положительными test-оценками; просмотренные '
        'train-фильмы исключены. NDCG учитывает порядок попаданий, Recall — долю найденных '
        'релевантных, Precision — их долю в десятке, MAP — точность на позициях попаданий. '
        'Покрытие — доля каталога хотя бы в одной рекомендации.')
    r.p(f'LightGCN − item-CF по NDCG: {number(difference["ndcg_difference"], 5)}, '
        f'95% парный интервал [{number(difference["ci95"][0], 5)}; {number(difference["ci95"][1], 5)}]. '
        'Интервал включает ноль, поэтому уверенного выигрыша GNN нет. Покрытие выше '
        '27,76% против 14,51%. Против popular рост NDCG заметен, но сложную модель '
        'нельзя выбирать только по названию архитектуры.')
    r.p('Выбран один слой, 200 эпох. Оба trials дошли до лимита: early stopping предусмотрен, '
        'но не сработал. Значит, найден лучший из проверенных вариантов, а не доказан '
        f'оптимум. Эксперимент занял {number(protocol["total_seconds"], 1)} с на CPU; '
        f'refit — {number(protocol["refit_seconds"], 2)} с. p95 готового recommend 0,541 мс '
        'не включает HTTP, загрузку checkpoint и конкурентную нагрузку.')
    r.p('Дополнительно модель восстановлена из state_dict: граф и embeddings совпали, '
        'метрики независимо пересчитаны для всех 915 пользователей с расхождением '
        'менее 1e-12. Протокол проверки — verification/block2-independent-review.md.')

    # 15 instructions
    r.page('Воспроизведение и ограничения')
    r.p('Команды выполняются из корня репозитория после установки зависимостей. '
        'Загрузка GGUF и бинарника производится отдельно, с проверкой размера и SHA256. '
        'Большие веса и runtime не включаются в git; небольшой checkpoint LightGCN сохранён.')
    r.code('source .venv/bin/activate\npip install -r multi_agent/requirements.txt\npython -m multi_agent.runtime.setup\npython -m recsys.download_data\npython -m recsys.evaluate_advanced\npython -m recsys.lightgcn --user 1 --k 5\npython -m multi_agent profile --user 1 --prefer Comedy --avoid Horror\npython -m multi_agent.sandbox -- -m multi_agent ask "Три комедии" --user 1\npython -m pytest -q', size=9)
    r.p('Для защищённого запуска используется multi_agent/sandbox.py. '
        'Точные команды и требования к macOS находятся в README '
        'модуля. Смена платформы требует другого бинарника llama.cpp и подходящей '
        'изоляции; архив macOS ARM64 нельзя считать универсальной установкой.')
    r.h('Что показано и что пока не проверено')
    r.p('Локальный прототип объединяет несколько LLM-ролей, фактические инструменты, '
        'память и проверку в один workflow. Он работает с каталогом MovieLens, а не '
        'с современным сервисом. Маленькие LLM могут неправильно понять русский '
        'запрос или принять слабое объяснение. JSON Schema не гарантирует верное '
        'содержание. Независимый валидатор тоже не является источником истины.')
    r.p('В E2E-проверке планировщик добавлял непрошенные жанры и неверно переносил '
        'запреты. Добавлен небольшой domain guard; исходные ответы LLM и benchmark '
        'не подменялись. Guard понимает короткие типовые фразы, но может ошибиться '
        'на сложных отрицаниях и цитатах. Полный конфликт желаемого и запрещённого '
        'жанра даёт no_results. Для нового пользователя evidence честно отмечает '
        'popular fallback вместо LightGCN.')
    r.p('Для LightGCN остаются холодный старт, смещение к популярным объектам и '
        'возможность продвижения фильмов подставными оценками. Контентные признаки '
        'помогают новым фильмам, но требуют новых данных; лимиты и доверие к аккаунтам '
        'могут задевать обычных активных пользователей. Атаки и защита от них '
        'экспериментально в этой работе не моделировались.')
    r.p('Нет нагрузочного теста, online-персонализации и A/B-теста на пользователях. '
        'Перед сервисным внедрением нужны свежий каталог, единый временной срез, '
        'согласование удаления логов, контроль обновления модели и план отката. '
        'Бизнес-гипотеза — больше полезных запусков из рекомендаций; проверять её '
        'следует по показам, стартам просмотра и удержанию, а не только по NDCG.')
    r.link('Исходники результаты и инструкции нового блока', BLOCK_REPO)

    # 16 defense and sources
    r.page('Подготовка к защите и источники')
    r.p('На демонстрации стоит открыть один настоящий run: показать план, ответ tool, '
        'черновик, решение проверяющего и итоговый статус. Затем изменить предпочтения, '
        'проверить инвалидацию кеша и показать негативный случай. Для LightGCN — '
        'объяснить одно распространение по графу и показать validation/test-разделение.')
    r.table(['Вопрос', 'Краткий ответ'], [
        ['LLM и агент', 'LLM генерирует текст; система агента добавляет состояние, выбор действий, tools и проверку'],
        ['Контекстное окно', 'Предел токенов запроса и ответа; большой контекст не гарантирует учёт всех фактов'],
        ['Галлюцинация', 'Неподтверждённое утверждение; ID и названия сверяются с evidence, текст проверяет отдельная модель'],
        ['Зачем другой family', 'Снижает зависимость от одинаковых ошибок; не отменяет детерминированные правила'],
        ['Почему не только RAG', 'Поиск похожего текста не гарантирует последнее значение ограничения и удаление памяти'],
        ['Что доказывают метрики', 'Качество фиксированного offline-набора; рост CTR и удержания требует online-теста'],
    ], [4.3, 12.5])
    r.h('Основные источники')
    references = [
        ('1 LangGraph — workflows and agents', 'https://docs.langchain.com/oss/python/langgraph/workflows-agents'),
        ('2 llama.cpp — официальный репозиторий', 'https://github.com/ggml-org/llama.cpp'),
        ('3 Qwen3.5 0.8B — model card', 'https://huggingface.co/Qwen/Qwen3.5-0.8B'),
        ('4 Qwen3.5 2B — model card', 'https://huggingface.co/Qwen/Qwen3.5-2B'),
        ('5 SmolLM3 3B — model card', 'https://huggingface.co/HuggingFaceTB/SmolLM3-3B'),
        ('6 C4 model — виды диаграмм', 'https://c4model.com/diagrams'),
        ('7 OpenTelemetry — сигналы', 'https://opentelemetry.io/docs/concepts/signals/'),
        ('8 Google Cloud — Two Tower retrieval', 'https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation'),
        ('9 Geng et al. P5 — RecSys 2022', 'https://arxiv.org/abs/2203.13366'),
        ('10 He et al. LightGCN — SIGIR 2020', 'https://arxiv.org/abs/2002.02126'),
        ('11 GroupLens — MovieLens 100K', 'https://grouplens.org/datasets/movielens/100k/'),
    ]
    for label, url in references:
        paragraph = r.link(label, url)
        paragraph.paragraph_format.space_after = Pt(3)
    r.p('Развёрнутое сравнение фреймворков, вариантов изоляции, памяти и наблюдаемости '
        'с дополнительными primary-ссылками находится в multi_agent/docs/research.md; '
        'вопросы защиты — в docs/defense.md. Дата сверки источников и запуска — 17.11.2026.')
    output = OUT / 'Block2_Bashirov_11-411.docx'
    r.doc.save(output)
    print(output)


if __name__ == '__main__':
    main()
