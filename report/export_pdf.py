"""Typeset the standalone report using ACL dimensions/fonts and vector figures.

The editable source embeds the official, unmodified ACL style for the native
LaTeX editor. Its compiler tool does not export PDF bytes, so this delivery path
reads the same prose/captions/bibliography and applies the published ACL layout.
Figure/equation PDFs are generated from validated source data by analyze_results.
"""
from pathlib import Path
import html
import os
import re
import sys
from functools import partial

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader, PdfWriter, Transformation

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / 'output/pdf/final_submission.pdf'
tex = (HERE / 'final_submission.tex').read_text()
body = tex.split(r'\begin{document}', 1)[1].split(r'\maketitle', 1)[1]
font_root = Path(os.environ.get('REPORT_FONT_DIR', '/System/Library/Fonts/Supplemental'))
for name, suffix in [('ReportSerif',''),('ReportSerif-Bold',' Bold'),('ReportSerif-Italic',' Italic'),('ReportSerif-BoldItalic',' Bold Italic')]:
    pdfmetrics.registerFont(TTFont(name, str(font_root / f'Times New Roman{suffix}.ttf')))
pdfmetrics.registerFontFamily('ReportSerif', normal='ReportSerif', bold='ReportSerif-Bold', italic='ReportSerif-Italic', boldItalic='ReportSerif-BoldItalic')
try:
    from matplotlib import get_data_path
    default_math = Path(get_data_path()) / 'fonts/ttf'
except ImportError:
    default_math = Path(sys.prefix).parent / 'native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype'
math_root = Path(os.environ.get('REPORT_MATH_FONT_DIR', str(default_math)))
for name,suffix in [('ReportMath',''),('ReportMath-Italic','-Italic')]:
    pdfmetrics.registerFont(TTFont(name,str(math_root/f'DejaVuSerif{suffix}.ttf')))
pdfmetrics.registerFontFamily('ReportMath',normal='ReportMath',italic='ReportMath-Italic',bold='ReportMath',boldItalic='ReportMath-Italic')

citations = {}
for label,key in re.findall(r'\\bibitem\[([^]]+)\]\{([^}]+)\}',body):
    author,year=label.replace('~',' ').split('(')
    citations[key]=(author,year.rstrip(')'))
refs={}
for environment in ('equation','figure*','table'):
    environments = r'table\*?' if environment == 'table' else re.escape(environment)
    for number,match in enumerate(re.finditer(r'\\begin\{('+environments+r')\}.*?\\end\{\1\}',body,re.S),1):
        for label in re.findall(r'\\label\{([^}]+)\}',match[0]):refs[label]=str(number)


def math_markup(value):
    value=value.replace(r'\ranks',r'\{1,4,8,16,32,64,128,256\}')
    value=re.sub(r'\\(?:mathrm|text)\{([^{}]+)\}',r'\1',value)
    value=re.sub(r'\\sqrt\{([^{}]+)\}',r'√(\1)',value)
    value=re.sub(r'\\sqrt\s*([A-Za-z])',r'√\1',value)
    replacements={'alpha':'α','beta':'β','theta':'θ','epsilon':'ε','Delta':'Δ','pm':'±','times':'×','in':'∈','approx':'≈','leq':'≤','qquad':'  ','quad':' '}
    value=re.sub(r'\\([A-Za-z]+)',lambda m:replacements.get(m[1],m[0]),value)
    value=value.replace(r'\{','LEFTBRACE').replace(r'\}','RIGHTBRACE').replace(r'\,',' ').replace(r'\ ',' ')
    for symbol,tag in [('_','sub'),('^','super')]:
        value=re.sub(re.escape(symbol)+r'\{([^{}]+)\}',rf'<{tag}>\1</{tag}>',value)
        value=re.sub(re.escape(symbol)+r'([A-Za-z0-9])',rf'<{tag}>\1</{tag}>',value)
    value=value.replace('{','').replace('}','').replace('LEFTBRACE','{').replace('RIGHTBRACE','}')
    if '\\' in value:raise ValueError('Unsupported inline math: '+value)
    value=value.replace(' ','\u00a0')
    return '<nobr><font name="ReportMath" size="10"><i>'+value+'</i></font></nobr>'


def markup(value):
    value=value.replace(r'\modelsmall{}','Llama-3.2-3B-Instruct').replace(r'\modelbase{}','Llama-3.1-8B-Instruct')
    value=html.escape(value,quote=False).replace(r"\'in",'ín')
    value=re.sub(r'\$([^$]+)\$',lambda m:math_markup(m[1]),value)
    def cite(m):
        kind,key=m.groups();author,year=citations[key]
        label=f'({author}, {year})' if kind=='p' else f'{author} ({year})'
        return f'<link href="#ref-{key}">{label}</link>'
    value=re.sub(r'\\cite([pt])\{([^}]+)\}',cite,value)
    value=re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}',lambda m:f'<link href="{m[1]}">{m[2]}</link>',value)
    for command,tag in [('emph','i'),('textbf','b')]:
        value=re.sub(r'\\'+command+r'\{([^{}]+)\}',rf'<{tag}>\1</{tag}>',value)
    value=re.sub(r'\\ref\{([^}]+)\}',lambda m:refs[m[1]],value)
    value=value.replace(r'\&amp;','&amp;').replace(r'\%','%').replace(r'\_','_')
    value=value.replace('~',' ').replace('---','—').replace('--','–').replace('``','“').replace("''",'”')
    value=re.sub(r'\s+',' ',value).strip()
    if '\\' in value:raise ValueError('Unsupported prose command: '+value)
    return value


# ACLPUB: A4, 2.5 cm margins, 0.6 cm column gap, 11 pt Times body.
W,H=A4
MARGIN=2.5/2.54*72
GAP=.6/2.54*72
TEXT_WIDTH=W-2*MARGIN
COLUMN=(TEXT_WIDTH-GAP)/2
BODY=ParagraphStyle('Body',fontName='ReportSerif',fontSize=11,leading=13.6,alignment=TA_JUSTIFY,firstLineIndent=11,spaceAfter=0,allowWidows=0,allowOrphans=0)
FIRST=ParagraphStyle('First',parent=BODY,firstLineIndent=0)
SECTION=ParagraphStyle('Section',parent=FIRST,fontName='ReportSerif-Bold',fontSize=12,leading=14.4,spaceBefore=8.5,spaceAfter=6.5,keepWithNext=True,alignment=TA_LEFT)
SUBSECTION=ParagraphStyle('Subsection',parent=SECTION,fontSize=11,leading=13.6,spaceBefore=7.5,spaceAfter=3.5)
ABSTRACT=ParagraphStyle('Abstract',parent=FIRST,fontSize=10,leading=12,leftIndent=.6/2.54*72,rightIndent=.6/2.54*72,spaceAfter=5)
CAPTION=ParagraphStyle('Caption',parent=FIRST,fontSize=10,leading=12,alignment=TA_LEFT)
CELL=ParagraphStyle('Cell',parent=CAPTION,spaceAfter=0)
REFERENCE=ParagraphStyle('Reference',parent=BODY,fontSize=10,leading=12,leftIndent=11,firstLineIndent=-11,alignment=TA_LEFT,spaceAfter=3)


def caption(block):
    # Captions in this report have no nested braces other than inline math.
    return re.search(r'\\caption\{(.*?)\}\s*\\label',block,re.S)[1]


figure_blocks=list(re.finditer(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',body,re.S))
figures=[]
for index,m in enumerate(figure_blocks,1):
    label=re.search(r'\\label\{([^}]+)\}',m[0])[1]
    name={'fig:size_alpha':'size_alpha','fig:larger':'larger'}[label]
    page=PdfReader(HERE/'figures'/f'{name}.pdf').pages[0]
    fig_height=float(page.mediabox.height)*TEXT_WIDTH/float(page.mediabox.width)
    cap=Paragraph(f'Figure {index}: '+markup(caption(m[0])),CAPTION)
    cap_height=cap.wrap(TEXT_WIDTH,H)[1]
    figures.append({'name':name,'height':fig_height,'caption':cap,'caption_height':cap_height,'reserved':fig_height+cap_height+14})

placements=[]

def on_page(canvas,doc,figure=None,wide_table=None):
    # ACL final mode uses neither line numbers nor printed page numbers.
    if doc.page==1:
        canvas.setFont('ReportSerif-Bold',15)
        canvas.drawCentredString(W/2,H-MARGIN-15,'Investigating Whether Inductive Backdoors')
        canvas.drawCentredString(W/2,H-MARGIN-33,'Are a LoRA Artifact')
        for y,names in [(H-MARGIN-61,('Harith Yerragolam','Sama Walke')),(H-MARGIN-103,('Sai Rithvik Achutuni','Srivatsav Reddy Poreddy'))]:
            for x,name in zip((W/2-103,W/2+103),names):
                canvas.setFont('ReportSerif-Bold',12);canvas.drawCentredString(x,y,name)
                canvas.setFont('ReportSerif',12);canvas.drawCentredString(x,y-15,'IIIT Hyderabad')
    if figure:
        y=H-MARGIN-figure['height']
        placements.append((doc.page,HERE/'figures'/(figure['name']+'.pdf'),MARGIN,y,TEXT_WIDTH,figure['height']))
        figure['caption'].drawOn(canvas,MARGIN,y-5-figure['caption_height'])
    if wide_table:
        y=H-MARGIN-wide_table['height']
        wide_table['table'].drawOn(canvas,MARGIN,y)
        wide_table['caption'].drawOn(canvas,MARGIN,y-5-wide_table['caption_height'])


def frames(top):
    return [Frame(MARGIN+i*(COLUMN+GAP),MARGIN,COLUMN,top-MARGIN,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id=f'column-{i}') for i in range(2)]


class Equation(Flowable):
    def __init__(self,name,number):
        super().__init__();self.path=HERE/'figures'/f'{name}.pdf';self.number=number
        page=PdfReader(self.path).pages[0]
        self.native_width=float(page.mediabox.width);self.native_height=float(page.mediabox.height)
        self.width=COLUMN;self.scale=min(1,(COLUMN-18)/self.native_width);self.height=self.native_height*self.scale+10
    def draw(self):
        x,y=self.canv.absolutePosition(0,0)
        width=self.native_width*self.scale;height=self.native_height*self.scale
        placements.append((self.canv.getPageNumber(),self.path,x+(COLUMN-18-width)/2,y+5,width,height))
        self.canv.setFont('ReportSerif',10)
        self.canv.drawRightString(COLUMN,self.height/2-3,f'({self.number})')


def table_content(block,width):
    inner=block.split(r'\toprule',1)[1].split(r'\bottomrule',1)[0].replace(r'\midrule','')
    rows=[]
    for row in inner.split(r'\\'):
        if row.strip():rows.append([Paragraph(markup(c.strip()),CELL) for c in row.split('&')])
    assert all(len(row)==4 for row in rows)
    weights=[.20,.57,.13,.10] if width>COLUMN else [.40,.22,.18,.20]
    table=Table(rows,colWidths=[width*weight for weight in weights])
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'ReportSerif'),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black)]))
    return table


def table_flowable(block,number):
    return KeepTogether([Spacer(1,7),table_content(block,COLUMN),Spacer(1,5),Paragraph(f'Table {number}: '+markup(caption(block)),CAPTION),Spacer(1,5)])


wide_tables=[]
for number,match in enumerate(re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}',body,re.S),1):
    if match[1]!='table*':continue
    table=table_content(match[0],TEXT_WIDTH)
    height=table.wrap(TEXT_WIDTH,H)[1]
    cap=Paragraph(f'Table {number}: '+markup(caption(match[0])),CAPTION)
    cap_height=cap.wrap(TEXT_WIDTH,H)[1]
    wide_tables.append({'table':table,'height':height,'caption':cap,'caption_height':cap_height,'reserved':height+cap_height+15})
assert len(wide_tables)==1


OUTPUT.parent.mkdir(parents=True,exist_ok=True)
class ACLDocTemplate(BaseDocTemplate):
    def afterPage(self):
        self.handle_nextPageTemplate(f'p{self.page+1}' if self.page < 8 else 'overflow')

doc=ACLDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=MARGIN,bottomMargin=MARGIN,initialFontName='ReportSerif',initialFontSize=11,initialLeading=13.6,title='Investigating Whether Inductive Backdoors Are a LoRA Artifact',author='Harith Yerragolam; Sama Walke; Sai Rithvik Achutuni; Srivatsav Reddy Poreddy')
# Full-width figures and design table sit at page tops. The native
# LaTeX preview chooses its own float positions; prose/data/captions are shared.
templates=[]
for page in range(1,9):
    figure={5:figures[0],6:figures[1]}.get(page)
    wide_table=wide_tables[0] if page==3 else None
    reserved=149.6 if page==1 else figure['reserved'] if figure else wide_table['reserved'] if wide_table else 0
    templates.append(PageTemplate(f'p{page}',frames(H-MARGIN-reserved),onPage=partial(on_page,figure=figure,wide_table=wide_table)))
templates.append(PageTemplate('overflow',frames(H-MARGIN),onPage=on_page))
doc.addPageTemplates(templates)
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',body,re.S)
story=[Paragraph('Abstract',ParagraphStyle('AbstractTitle',parent=SECTION,alignment=TA_CENTER,spaceBefore=0,spaceAfter=6))]
story.append(Paragraph(markup(abstract[1]),ABSTRACT))
main=body[abstract.end():].split(r'\begin{thebibliography}',1)[0]
pattern=r'(\\section\*?\{[^}]+\}|\\subsection\{[^}]+\}|\\begin\{equation\}.*?\\end\{equation\}|\\begin\{figure\*\}.*?\\end\{figure\*\}|\\begin\{table\*?\}.*?\\end\{table\*?\})'
section=subsection=equation=table_number=0
after_heading=True
for block in re.split(pattern,main,flags=re.S):
    if not block.strip():continue
    if block.startswith(r'\section'):
        title=re.search(r'\{([^}]+)\}',block)[1]
        if not block.startswith(r'\section*'):
            section+=1;subsection=0;title=f'{section}  '+title
        story.append(Paragraph(markup(title),SECTION));after_heading=True
    elif block.startswith(r'\subsection'):
        subsection+=1;title=re.search(r'\{([^}]+)\}',block)[1]
        story.append(Paragraph(f'{section}.{subsection}  '+markup(title),SUBSECTION));after_heading=True
    elif block.startswith(r'\begin{figure*}'):
        continue
    elif block.startswith(r'\begin{equation}'):
        equation+=1;story.append(Equation('equation_gap' if 'g_{' in block else 'equation_scale',equation))
    elif block.startswith((r'\begin{table}',r'\begin{table*}')):
        table_number+=1
        if block.startswith(r'\begin{table}'):
            story.append(table_flowable(block,table_number))
    else:
        for part in re.split(r'\n\s*\n',block.strip()):
            if part.strip():
                story.append(Paragraph(markup(part),FIRST if after_heading else BODY));after_heading=False
story.append(Paragraph('References',SECTION))
bib=body.split(r'\begin{thebibliography}{11}',1)[1].split(r'\end{thebibliography}',1)[0]
for key,entry in re.findall(r'\\bibitem\[[^]]+\]\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)',bib,re.S):
    story.append(Paragraph(f'<a name="ref-{key}"/>'+markup(entry),REFERENCE))
doc.build(story,canvasmaker=partial(Canvas,initialFontName='ReportSerif',initialFontSize=11,initialLeading=13.6))
reader=PdfReader(OUTPUT)
assert len(reader.pages)<=8,f'Over eight-page total: {len(reader.pages)}'
assert {str(p[1].name) for p in placements if not p[1].name.startswith('equation')}=={'size_alpha.pdf','larger.pdf'}
writer=PdfWriter()
vector_readers={path:PdfReader(path) for _,path,*_ in placements}
for number,page in enumerate(reader.pages,1):
    for target,path,x,y,width,height in placements:
        if target!=number:continue
        vector=vector_readers[path].pages[0]
        transform=Transformation().scale(width/float(vector.mediabox.width),height/float(vector.mediabox.height)).translate(x,y)
        page.merge_transformed_page(vector,transform,over=True)
    writer.add_page(page)
writer.add_metadata({'/Title':'Investigating Whether Inductive Backdoors Are a LoRA Artifact','/Author':'Harith Yerragolam; Sama Walke; Sai Rithvik Achutuni; Srivatsav Reddy Poreddy','/Subject':'LoRA generalization; ACL final format'})
tmp=OUTPUT.with_suffix('.tmp.pdf')
with tmp.open('wb') as handle:writer.write(handle)
tmp.replace(OUTPUT)
check=PdfReader(OUTPUT)
plain='\n'.join(page.extract_text() for page in check.pages)
assert '\x00' not in plain and '??' not in plain
embedded_fonts=set()
for page in check.pages:
    assert abs(float(page.mediabox.width)-W)<.01 and abs(float(page.mediabox.height)-H)<.01
    for font_ref in page['/Resources'].get('/Font',{}).values():
        font=font_ref.get_object()
        descendants=font.get('/DescendantFonts',[font])
        for descendant_ref in descendants:
            descendant=descendant_ref.get_object()
            descriptor=descendant.get('/FontDescriptor')
            assert descriptor is not None, ('missing font descriptor',font)
            descriptor=descriptor.get_object()
            assert any(key in descriptor for key in ('/FontFile','/FontFile2','/FontFile3')), ('unembedded font',font)
        embedded_fonts.add(str(font['/BaseFont']))
print(f'Exported {len(check.pages)} A4 pages; all {len(embedded_fonts)} fonts embedded; vector figures: {OUTPUT}')
