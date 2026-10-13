"""Схемы реальных модулей и HTTP-запроса. Требуется Pillow; PNG и редактируемый SVG."""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
INK, EDGE, FILL = '#172b40', '#748599', '#edf2f7'
FONT_CANDIDATES = [
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/arial.ttf',
]
FONT = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
if FONT is None:
    raise RuntimeError('Установите Arial или DejaVu Sans с кириллицей')

class Diagram:
    def __init__(self,w,h):
        self.w,self.h,self.scale=w,h,2
        self.im=Image.new('RGB',(w*2,h*2),'white');self.draw=ImageDraw.Draw(self.im)
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                  '<rect width="100%" height="100%" fill="white"/>']
    def text(self,x,y,text,size=23):
        lines=text.split('\n'); font=ImageFont.truetype(FONT,size*2)
        for i,line in enumerate(lines):
            cy=y+(i-(len(lines)-1)/2)*size*1.3
            self.draw.text((x*2,cy*2),line,font=font,fill=INK,anchor='mm')
            self.svg.append(f'<text x="{x}" y="{cy}" text-anchor="middle" dominant-baseline="central" font-family="Arial,DejaVu Sans,sans-serif" font-size="{size}" fill="{INK}">{escape(line)}</text>')
    def box(self,x,y,w,h,label,size=23):
        self.draw.rounded_rectangle((x*2,y*2,(x+w)*2,(y+h)*2),radius=9,fill=FILL,outline=EDGE,width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{FILL}" stroke="{EDGE}"/>')
        self.text(x+w/2,y+h/2,label,size)
    def arrow(self,a,b):
        x,y=b; angle=math.atan2(y-a[1],x-a[0]); points=[b]
        for sign in [-1,1]:points.append((x-10*math.cos(angle+sign*.48),y-10*math.sin(angle+sign*.48)))
        self.draw.line([tuple(v*2 for v in p) for p in [a,b]],fill=EDGE,width=3)
        self.draw.polygon([tuple(v*2 for v in p) for p in points],fill=EDGE)
        self.svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{x}" y2="{y}" stroke="{EDGE}" stroke-width="1.5"/>')
        self.svg.append(f'<polygon points="{" ".join(f"{px},{py}" for px,py in points)}" fill="{EDGE}"/>')
    def save(self,name):
        self.im.save(OUT/f'{name}.png')
        (OUT/f'{name}.svg').write_text('\n'.join(self.svg+['</svg>']))

d=Diagram(1000,590)
for x,label in [(140,'Входные данные'),(500,'Модуль проекта'),(860,'Результат')]:d.text(x,30,label,25)
rows=[('Текст отзыва','text / RuBERT','Метка тональности'),('Аудиофайл WAV','audio / YAMNet','Класс звука'),
      ('Скриншот сайта','image / CLIP','Тип страницы'),('Видеофайл','video / YOLO11','Видео с рамками'),
      ('Оценки MovieLens','recsys / 5 методов','Топ-10 фильмов'),('Текст запроса','llm / Qwen','Текст ответа')]
for i,(a,b,c) in enumerate(rows):
    y=70+i*85
    d.box(10,y,260,60,a,22);d.box(350,y,300,60,b);d.box(730,y,260,60,c,22)
    d.arrow((275,y+30),(343,y+30));d.arrow((655,y+30),(723,y+30))
d.save('architecture_modules')

d=Diagram(1000,335)
d.box(10,130,195,85,'Клиент\nPOST /predict',22)
d.box(270,130,205,85,'FastAPI\nпроверка файла',22)
d.box(540,130,200,85,'predict(image)\nCLIP + lock',22)
d.box(805,130,185,85,'JSON\nPrediction',22)
for a,b in [(210,263),(480,533),(745,798)]:d.arrow((a,172),(b,172))
d.box(270,265,205,50,'400 / 413 / 415',20)
d.arrow((372,220),(372,258))
d.box(520,15,240,60,'Кеш весов модели',22);d.arrow((640,80),(640,123))
d.text(640,292,'Ошибка модели → 503',19)
d.text(895,48,'label, score,\nprobabilities, model',20)
d.text(108,48,'PNG / JPEG / WebP',20)
d.save('architecture_api')
print('Сохранены две схемы: PNG и SVG.')
