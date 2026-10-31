"""Export this report's LaTeX content to a standalone two-column PDF.

The native Codex compiler validates/previews final_submission.tex but its tool
does not export PDF bytes. This narrow ReportLab renderer reads that same source;
it is not a general LaTeX compiler. It rejects unsupported residual commands.
"""
from pathlib import Path
import html
import re
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "final_submission.tex"
OUTPUT = HERE.parent / "output/pdf/final_submission.pdf"
FONT_ROOT = Path(os.environ.get("REPORT_FONT_DIR", "/System/Library/Fonts/Supplemental"))
for name, suffix in [("ReportSerif", ""), ("ReportSerif-Bold", " Bold"),
                     ("ReportSerif-Italic", " Italic"), ("ReportSerif-BoldItalic", " Bold Italic")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / f"Times New Roman{suffix}.ttf")))
pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportSerif-Bold",
                              italic="ReportSerif-Italic", boldItalic="ReportSerif-BoldItalic")
try:
    from matplotlib import get_data_path
    default_math_root = Path(get_data_path()) / "fonts/ttf"
except ImportError:
    # The desktop document runtime also bundles these fonts with LibreOffice.
    default_math_root = Path(sys.prefix).parent / "native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype"
math_root = Path(os.environ.get("REPORT_MATH_FONT_DIR", str(default_math_root)))
for name, suffix in [("ReportMath", ""), ("ReportMath-Italic", "-Italic")]:
    pdfmetrics.registerFont(TTFont(name, str(math_root / f"DejaVuSerif{suffix}.ttf")))
pdfmetrics.registerFontFamily("ReportMath", normal="ReportMath", italic="ReportMath-Italic",
                              bold="ReportMath", boldItalic="ReportMath-Italic")

tex = SOURCE.read_text()
citations = {}
for label, key in re.findall(r"\\bibitem\[([^]]+)\]\{([^}]+)\}", tex):
    author, year = label.replace("~", " ").split("(")
    citations[key] = (author, year.rstrip(")"))


def math_markup(value):
    value = value.replace(r"\ranks", r"\{1,4,8,16,32,64,128,256\}")
    value = value.replace(r"\mathbb{R}", "ℝ")
    value = re.sub(r"\\(?:mathrm|text)\{([^{}]+)\}", r"\1", value)
    value = value.replace(r"\frac{1}{3}", "⅓")
    value = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", value)
    value = re.sub(r"\\sqrt\s*([A-Za-z])", r"√\1", value)
    replacements = {"alpha": "α", "beta": "β", "theta": "θ", "epsilon": "ε",
                    "times": "×", "in": "∈", "approx": "≈", "sum": "Σ",
                    "qquad": "  ", "quad": " ", "mathrm": ""}
    value = re.sub(r"\\([A-Za-z]+)", lambda m: replacements.get(m[1], m[0]), value)
    value = value.replace(r"\{", "LEFTBRACE").replace(r"\}", "RIGHTBRACE")
    value = value.replace(r"\ ", " ").replace(r"\,", " ").replace(r"\!", "")
    for symbol, tag in [("_", "sub"), ("^", "super")]:
        value = re.sub(re.escape(symbol) + r"\{([^{}]+)\}", rf"<{tag}>\1</{tag}>", value)
        value = re.sub(re.escape(symbol) + r"([A-Za-z0-9])", rf"<{tag}>\1</{tag}>", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("LEFTBRACE", "{").replace("RIGHTBRACE", "}")
    if "\\" in value:
        raise ValueError(f"Unsupported math: {value}")
    value = value.replace("ℝ", "</i>ℝ<i>")
    return f'<font name="ReportMath" size="9"><i>{value}</i></font>'


def markup(value):
    value = value.replace(r"\modelsmall{}", "Llama-3.2-3B-Instruct")
    value = value.replace(r"\modelbase{}", "Llama-3.1-8B-Instruct")
    value = html.escape(value, quote=False)
    value = value.replace(r"\'in", "ín")
    value = re.sub(r"\$([^$]+)\$", lambda m: math_markup(m[1]), value)
    def cite(match):
        kind, key = match.groups()
        author, year = citations[key]
        text = f"{author}, {year}" if kind == "p" else f"{author} ({year})"
        if kind == "p":
            text = f"({text})"
        return f'<link href="#ref-{key}">{text}</link>'
    value = re.sub(r"\\cite([pt])\{([^}]+)\}", cite, value)
    value = re.sub(r"\\href\{([^}]+)\}\{([^}]+)\}",
                   lambda m: f'<link href="{m[1]}">{m[2]}</link>', value)
    for command, tag in [("emph", "i"), ("textbf", "b")]:
        value = re.sub(r"\\" + command + r"\{([^{}]+)\}", rf"<{tag}>\1</{tag}>", value)
    value = re.sub(r"\\texttt\{([^{}]+)\}", r'<font name="Courier" size="8">\1</font>', value)
    value = re.sub(r"\\ref\{(?:eq:scale|tab:designs)\}", "1", value)
    value = value.replace(r"\&amp;", "&amp;").replace(r"\_", "_")
    value = value.replace("~", " ").replace("---", "—").replace("--", "–")
    value = value.replace("``", "“").replace("''", "”")
    value = re.sub(r"\s+", " ", value).strip()
    if "\\" in value:
        raise ValueError(f"Unsupported command: {value}")
    return value


BODY = ParagraphStyle("Body", fontName="ReportSerif", fontSize=10, leading=11.7,
                      alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=3,
                      allowWidows=0, allowOrphans=0)
SECTION = ParagraphStyle("Section", parent=BODY, fontName="ReportSerif-Bold", fontSize=11.5,
                         leading=13, firstLineIndent=0, spaceBefore=9, spaceAfter=5,
                         keepWithNext=True, alignment=TA_LEFT)
SUBSECTION = ParagraphStyle("Subsection", parent=SECTION, fontSize=10, leading=11.8,
                            spaceBefore=7, spaceAfter=4)
ABSTRACT = ParagraphStyle("Abstract", parent=BODY, leftIndent=9, rightIndent=9,
                          fontSize=9.5, leading=11, firstLineIndent=0, spaceAfter=8)
CENTER = ParagraphStyle("Center", parent=BODY, alignment=TA_CENTER, firstLineIndent=0)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.8, leading=10.2,
                         firstLineIndent=0, spaceAfter=6, alignment=TA_LEFT)
CELL = ParagraphStyle("Cell", parent=CAPTION, fontSize=8.5, leading=9.8, spaceAfter=0)
REFERENCE = ParagraphStyle("Reference", parent=BODY, fontSize=9, leading=10.5,
                           firstLineIndent=-8, leftIndent=8, alignment=TA_LEFT, spaceAfter=6)
NOTE = ParagraphStyle("Note", parent=BODY, firstLineIndent=0, alignment=TA_LEFT)


def paragraphs(text, style=BODY):
    return [Paragraph(markup(part), style) for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def design_table(block, width):
    # Read every table cell directly from the canonical LaTeX source.
    inner = block.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]
    inner = inner.replace(r"\midrule", "")
    rows = []
    for row in inner.split(r"\\"):
        if not row.strip():
            continue
        cells = [cell.strip() for cell in row.split("&")]
        if len(cells) != 4:
            raise ValueError(f"Unexpected table row: {cells}")
        # Compact four original columns into three, retaining seeds with factors.
        rows.append([Paragraph(markup(cells[0]), CELL),
                     Paragraph(markup(cells[1] + ("; seeds: " + cells[2] if rows else "; seeds")), CELL),
                     Paragraph(markup(cells[3]), CELL)])
    table = Table(rows, colWidths=[width * .28, width * .56, width * .16])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), .4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black),
    ]))
    caption = re.search(r"\\caption\{(.*?)\}\s*\\label", block, re.S)[1]
    return KeepTogether([Spacer(1, 6), table, Spacer(1, 5), Paragraph("Table 1: " + markup(caption), CAPTION)])


width, height = A4
margin, gap = 66.61, 18.43  # 2.35 cm / 0.65 cm, matching the editable source.
column = (width - 2 * margin - gap) / 2


def page_header(canvas, doc):
    canvas.setFont("ReportSerif", 9)
    canvas.drawCentredString(width / 2, 39, str(doc.page))
    if doc.page != 1:
        return
    canvas.setFont("ReportSerif-Bold", 16)
    canvas.drawCentredString(width / 2, height - 72, "Investigating Whether Inductive Backdoors")
    canvas.drawCentredString(width / 2, height - 91, "Are a LoRA Artifact")
    canvas.setFont("ReportSerif", 11)
    centers = [width / 2 - 103, width / 2 + 103]
    for y, names in [(height - 122, ["Harith Yerragolam", "Sama Walke"]),
                     (height - 162, ["Sai Rithvik Achutuni", "Srivatsav Reddy Poreddy"])]:
        for x, name in zip(centers, names):
            canvas.drawCentredString(x, y, name)
            canvas.drawCentredString(x, y - 14, "IIIT Hyderabad")


def frames(top):
    return [Frame(margin + i * (column + gap), margin, column, top - margin,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id=f"col-{i}") for i in range(2)]


doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin,
                      title="Investigating Whether Inductive Backdoors Are a LoRA Artifact",
                      author="Harith Yerragolam; Sama Walke; Sai Rithvik Achutuni; Srivatsav Reddy Poreddy")
doc.addPageTemplates([
    PageTemplate("first", frames(height - 203), onPage=page_header, autoNextPageTemplate="later"),
    PageTemplate("later", frames(height - margin), onPage=page_header),
])

body = tex.split(r"\maketitle", 1)[1].split(r"\begin{thebibliography}", 1)[0]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
story = [Paragraph("Abstract", ParagraphStyle("AbstractTitle", parent=SUBSECTION, alignment=TA_CENTER)),
         *paragraphs(abstract[1], ABSTRACT)]
body = body[abstract.end():]
pattern = r"(\\section\{[^}]+\}|\\subsection\{[^}]+\}|\\begin\{equation\}.*?\\end\{equation\}|\\begin\{table\*\}.*?\\end\{table\*\})"
section = subsection = equation = 0
reserved_start = None
for block in re.split(pattern, body, flags=re.S):
    if not block.strip():
        continue
    if block.startswith(r"\section{"):
        section += 1
        subsection = 0
        if section == 5:
            reserved_start = len(story)
        story.append(Paragraph(f"{section}  " + markup(block[9:-1]), SECTION))
    elif block.startswith(r"\subsection{"):
        subsection += 1
        story.append(Paragraph(f"{section}.{subsection}  " + markup(block[12:-1]), SUBSECTION))
    elif block.startswith(r"\begin{equation}"):
        equation += 1
        equations = [
            '<i>α<sub>c</sub>(r) = c√(r/32),   s<sub>c</sub>(r) = α<sub>c</sub>(r)/√r = c/√32.</i>   (1)',
            '<i>g<sub>θ,j</sub>(y) = p<sub>θ,j</sub>(y) − ⅓ Σ<super>2026</super><sub>z=2024</sub> p<sub>θ,j</sub>(z).</i>   (2)',
        ]
        story.append(Paragraph('<font name="ReportMath">' + equations[equation - 1] + '</font>', ParagraphStyle("Equation", parent=CENTER, fontSize=8.2, leading=19, spaceBefore=4, spaceAfter=4)))
    elif block.startswith(r"\begin{table*}"):
        story.append(design_table(block, column))
    else:
        story.extend(paragraphs(block, NOTE if section >= 5 else BODY))

if reserved_start is not None:
    story[reserved_start:] = [KeepTogether(story[reserved_start:])]
story.append(Paragraph("References", SECTION))
bib = tex.split(r"\begin{thebibliography}{11}", 1)[1].split(r"\end{thebibliography}", 1)[0]
for key, entry in re.findall(r"\\bibitem\[[^]]+\]\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)", bib, re.S):
    story.append(Paragraph(f'<a name="ref-{key}"/>' + markup(entry), REFERENCE))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
reader = PdfReader(OUTPUT)
if len(reader.pages) > 8:
    raise ValueError(f"Report exceeds eight-page cap: {len(reader.pages)} pages")
plain = "\n".join(page.extract_text() for page in reader.pages)
if "■" in plain or "??" in plain or "\x00" in plain:
    raise ValueError("Missing glyph or unresolved reference in exported PDF")
print(f"Exported {len(reader.pages)} pages to {OUTPUT}")
