"""Render the D11 local-run review. Reads JSON evidence; never executes a strategy."""
from pathlib import Path
import html
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.d11_review_results import review


def build():
    review()
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
    from reportlab.lib.pagesizes import A4
    font='D11ReviewSans'
    pdfmetrics.registerFont(TTFont(font,str(ROOT/'reports/d11_review_assets/D11ReviewSans.ttf')))
    pdfmetrics.registerFontFamily(font,normal=font,bold=font,italic=font,boldItalic=font)
    styles={
        'body':ParagraphStyle('body',fontName=font,fontSize=9.5,leading=15,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor('#243448')),
        'title':ParagraphStyle('title',fontName=font,fontSize=20,leading=28,spaceAfter=15,wordWrap='CJK',textColor=colors.HexColor('#143a48')),
        'h2':ParagraphStyle('h2',fontName=font,fontSize=16,leading=24,spaceAfter=14,wordWrap='CJK',keepWithNext=True,textColor=colors.HexColor('#17675c')),
        'cell':ParagraphStyle('cell',fontName=font,fontSize=8.5,leading=13,wordWrap='CJK'),
        'header':ParagraphStyle('header',fontName=font,fontSize=8.5,leading=13,wordWrap='CJK',textColor=colors.white),
        'code':ParagraphStyle('code',fontName='Courier',fontSize=8.2,leading=13,spaceAfter=4),
    }
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<link href="{m[2]}" color="#17675c">{m[1]}</link>',value)
        value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
        return re.sub(r'`([^`]+)`',r'<font color="#17675c">\1</font>',value)
    content=(ROOT/'reports/D11研究报告.md').read_text(encoding='utf-8')
    lines=content.splitlines();story=[];i=0;code=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):code=not code;i+=1;continue
        if code:story.append(Paragraph(html.escape(line),styles['code']));i+=1;continue
        if line=='<!-- PAGEBREAK -->':story.append(PageBreak());i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['title']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in row):rows.append(row)
                i+=1
            n=len(rows[0]);widths={2:[.52,.48],3:[.29,.33,.38],4:[.35,.21,.21,.23],5:[.17,.14,.17,.25,.27]}[n]
            cells=[[Paragraph(inline(x),styles['header' if j==0 else 'cell']) for x in row] for j,row in enumerate(rows)]
            table=Table(cells,colWidths=[499*w for w in widths],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1d5360')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,10)]);continue
        story.append(Paragraph(inline(line),styles['body']));i+=1
    def page(canvas,doc):
        w,h=A4;canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d7e2e4'))
        canvas.line(48,h-37,w-48,h-37);canvas.line(48,36,w-48,36)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526979'))
        canvas.drawString(48,h-28,'CSI500 / D11 / LOCAL RUN REVIEW / 2026-09-06')
        canvas.drawRightString(w-48,h-28,'d11_v1 | source 51c280f')
        canvas.drawString(48,24,'Post-OOS diagnostic | JSON evidence reviewed | Historical No-Go retained')
        canvas.drawRightString(w-48,24,str(doc.page));canvas.restoreState()
    path=ROOT/'reports/D11研究报告.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=54,bottomMargin=49,title='中证500 D11研究报告：本机运行结果更新版',author='CSI500 Research')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(path)


if __name__=='__main__':build()
