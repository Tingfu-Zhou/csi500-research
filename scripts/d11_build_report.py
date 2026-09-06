from pathlib import Path
import json, csv, hashlib, shutil, re, html, math
from urllib.parse import quote
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4

import argparse, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.d11_io import D11Output
parser=argparse.ArgumentParser(description='Rebuild the archived D11 research report, not repaired P&L.')
parser.add_argument('--run-id',required=True)
args=parser.parse_args()
OUT=D11Output(ROOT,args.run_id).path
ARCHIVE=ROOT/'reports/d11_research'
(OUT/'figures').mkdir(exist_ok=True)
E=ARCHIVE/'sources'
SHA='42c8870392a3d45f531cffbefbc138e5d8fbb328'
BASE=f'https://github.com/Tingfu-Zhou/csi500-research/blob/{SHA}/'
read=lambda p:json.loads((E/p).read_text())
s=read('reports/d9_v2_summary.json'); d=read('reports/d10_summary.json')
p=s['performance']
def pct(x,n=2):return f'{x*100:.{n}f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

gates=table(['指标','实测','原门槛','事后对照'],[
['费后年化超额（月频）',pct(p['annualized_excess_return']),'≥3.5%','未通过'],
['信息比率IR（月频年化）',f"{p['information_ratio']:.3f}",'≥0.75','未通过'],
['Newey-West t（lag 3-6最低）',f"{p['newey_west_t_minimum']:.3f}",'≥2','未通过'],
['最大相对回撤（日频净值比）',pct(p['maximum_relative_drawdown']),'≤10%','未通过'],
['费后特异性贡献占比',pct(d['attribution']['oos']['specific_share_net']),'≥50%','按现有归因约定通过'],
['冲击系数翻倍后年化超额',pct(s['impact_2x']['annualized_excess_return']),'>0','通过']])
method=table(['项目','本次实际口径'],[
['研究/绩效窗口','设计2014-2018；设计组合有效信号2017-2018；样本外2019-01至2026-08'],
['股票池/基准','PIT中证500及登记的成分修正；index_weight为权重来源；H00905.CSI全收益为绩效基准'],
['特征/预测器','Qlib 0.9.7 Alpha158，158列；LightGBM 4.7.0 LGBMRegressor默认参数，random_state=42'],
['标签/训练','T+1 VWAP至T+6 VWAP的5日收益减同窗基准篮子收益；周截面标签z-score；按年扩展重训'],
['校准/切分','预测截面排序正态化；用此前已实现走向前预测估计收益系数；训练标签跨界purge'],
['决策/撮合','配置为T日收盘出信号、T+1全天VWAP成交；目标w₀时点的继承问题另见局限'],
['优化约束','多头满仓；个股主动偏离±1.5%；行业±2%；Size/Beta±0.3σ；V2主动单边换手≤15%'],
['费用/交易','线性费用进入LP；动态冲击在撮合计费；停牌/涨跌停规则与5%当日成交额上限；主资金1000万'],
['线性费率','佣金0.015%、过户费0.001%；卖出印花税2023-08-28前0.1%、之后0.05%；为冻结假设'],
['模型目标','最大化主动权重×校准预期超额－线性换手费用；Σ仅用于报告事前TE']])
yearly=table(['年度','Rank IC','月频年化超额','IR','NW-t','相对回撤'],[[str(r['prediction_year'])+('（1-8月）' if r['prediction_year']==2026 else ''),f"{r['rank_ic_mean']:.4f}",pct(r['annualized_excess_return']),f"{r['information_ratio']:.3f}",f"{r['newey_west_t']:.3f}",pct(r['maximum_relative_drawdown'])] for r in s['yearly']])
gross=[10.46,10.46,7.92,1.15,-0.38,5.25,5.30]
cost=[0,7.38,6.60,11.47,11.17,2.83,1.68]
names=['Top100理想基线','加入真实费用','加入可交易性','加入行业约束（诊断）','加入Size/Beta（诊断）','加入主动换手（诊断）','完整约束最终层']
ladder=table(['层','情景','净超额','费前诊断','成本拖累'],[[f'L{i}',names[i],pct(s['oos_ladder'][i]['annualized_excess_return']),f'{gross[i]:.2f}%',f'{cost[i]:.2f}pp'] for i in range(7)])
a=d['attribution']
attribution=table(['项目','设计期','样本外'],[[name,f"{a['design']['annualized_contributions'][key]*100:+.4f}",f"{a['oos']['annualized_contributions'][key]*100:+.4f}"] for key,name in [('industry','行业'),('style','风格'),('specific_net','特异性（费后）'),('benchmark_bridge','官方指数桥接'),('excess_return','净超额合计')]])
stress=table(['情景','年化超额','IR','NW-t最低','相对回撤'],[['D9-V2主结果',pct(p['annualized_excess_return']),f"{p['information_ratio']:.3f}",f"{p['newey_west_t_minimum']:.3f}",pct(p['maximum_relative_drawdown'])]]+[[name,pct(r['annualized_excess_return']),f"{r['information_ratio']:.3f}",f"{r['newey_west_t_minimum']:.3f}",pct(r['maximum_relative_drawdown'])] for name,r in zip(['冲击系数×2（复用）','延迟一天（冻结目标）','双周（冻结目标）'],d['stress_scenarios'])])
july=table(['2026年7月','当月累计收益'],[[name,pct(d['july_2026'][key])] for name,key in [('模型绝对收益','portfolio_return'),('H00905.CSI全收益指数','benchmark_return'),('超额（两者之差）','excess_return')]])

sources=[
('S01','reports/d9_v2_summary.json','performance / yearly / oos_ladder / model / constraints'),
('S02','reports/d10_summary.json','attribution / stress_scenarios / july_2026 / diagnostic_acceptance'),
('S03','reports/D10归因与诊断报告.md','§02阶梯原表、§01诊断、§10本机证据范围'),
('S04','D10勘误登记.md','E01-E06及C01-C06'),
('S05','reports/d4_summary.json','model.parameters / validation_metrics'),
('S06','config/config.yaml','d6 / d7 / training / costs'),
('S07','src/d9_core.py','aggregate_daily_to_monthly / maximum_relative_drawdown / summarize_monthly_performance'),
('S08','src/d8_core.py','summarize_backtest_performance（阶梯CAGR口径）'),
('S09','reports/d9_v2_manifest.json','原始/恢复manifest引用及可行性审计引用'),
('S10','scripts/d9_pipeline.py','build_model_dataset；配合scripts/d7_pipeline.py holding_returns_between'),
]
source_table=table(['引用','固定提交下来源','定位字段/内容'],[[sid,f'[{path}]({BASE}{quote(path)})',desc] for sid,path,desc in sources])
content=(ARCHIVE/'report_template.md').read_text()
for key,value in dict(gates=gates,method_table=method,yearly=yearly,ladder=ladder,attribution=attribution,stress=stress,july=july,source_table=source_table).items():content=content.replace('{{'+key+'}}',value)
assert '{{' not in content
(OUT/'D11研究报告.md').write_text(content)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.8,3.0))
ys=[r['annualized_excess_return']*100 for r in s['yearly']]
ax.bar(range(8),ys,color=['#b54738' if v<0 else '#27786b' for v in ys],width=.58)
ax.axhline(0,color='#334155',lw=.7);ax.axhline(3.5,color='#bd8523',ls='--',lw=1,label='Reference hurdle: 3.5%')
ax.set_xticks(range(8),[str(r['prediction_year'])+('*' if r['prediction_year']==2026 else '') for r in s['yearly']]);ax.set_ylim(-9.4,11.5)
for i,y in enumerate(ys):ax.text(i,y+(.35 if y>=0 else -.5),f'{y:.2f}%',ha='center',va='bottom' if y>=0 else 'top',fontsize=10)
ax.set_ylabel('Monthly-mean annualized excess (%)');ax.legend(frameon=False,loc='upper left',fontsize=9)
fig.text(.13,.005,'D9-V2 post-hoc amendment | Net of costs | *2026: January-August, annualized',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'figures/d11_yearly.png',dpi=180);plt.close(fig)
for img in (E/'reports/figures').glob('*.png'):shutil.copy2(img,OUT/'figures'/img.name)

# Preserve exact source bytes and make file-level and number-level evidence indices.
shutil.copytree(E,OUT/'sources',dirs_exist_ok=True)
(OUT/'plans').mkdir(exist_ok=True)
for f in (ARCHIVE/'plans').glob('*.md'):shutil.copy2(f,OUT/'plans'/f.name)
baseline_tree=json.loads((ROOT/'config/d11_baseline.json').read_text())['protected_blobs'];tree={p:{'sha':sha} for p,sha in baseline_tree.items()}
hashrows=[]; mismatches=[]
for f in sorted(E.rglob('*')):
 if not f.is_file():continue
 rel=f.relative_to(E).as_posix();b=f.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 assert blob==tree[rel]['sha'],rel
 hashrows.append({'local_path':'sources/'+rel,'repository_path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':blob,'git_commit':SHA,'source_url':BASE+quote(rel)})
for f in sorted((OUT/'plans').glob('*')):
 b=f.read_bytes();hashrows.append({'local_path':'plans/'+f.name,'repository_path':'','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':'','git_commit':'','source_url':'user attachment'})
with (OUT/'D11_sources_sha256.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(hashrows[0]));w.writeheader();w.writerows(hashrows)

# Locate each JSON scalar in the original, pretty-printed source using its key and exact value.
numrows=[]
def walk(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items():yield from walk(v,path+'/'+str(k).replace('~','~0').replace('/','~1'))
 elif isinstance(obj,list):
  for i,v in enumerate(obj):yield from walk(v,path+'/'+str(i))
 elif isinstance(obj,(int,float,bool)) or obj is None:yield path,obj
for f in sorted((E/'reports').glob('*.json')):
 text=f.read_text();lines=text.splitlines();rel=f.relative_to(E).as_posix()
 for pointer,value in walk(json.loads(text)):
  key=pointer.rsplit('/',1)[-1].replace('~1','/').replace('~0','~');needle=json.dumps(key,ensure_ascii=False)+': '+json.dumps(value,ensure_ascii=False)
  matches=[str(i+1) for i,l in enumerate(lines) if needle in l]
  first=matches[0] if matches else ''
  numrows.append({'source':rel,'locator':pointer,'exact_value':json.dumps(value,ensure_ascii=False),'line_candidates':';'.join(matches),'source_url':BASE+quote(rel)+(f'#L{first}' if first else '')})
for f in sorted((E/'reports').glob('*.md')):
 for i,line in enumerate(f.read_text().splitlines(),1):
  if re.search(r'\d',line):
   rel=f.relative_to(E).as_posix();numrows.append({'source':rel,'locator':f'line {i}','exact_value':line,'line_candidates':str(i),'source_url':BASE+quote(rel)+f'#L{i}'})
with (OUT/'D11_numeric_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(numrows[0]));w.writeheader();w.writerows(numrows)

manifest=read('reports/d9_v2_manifest.json')
checks={}
for k in ['original_manifest','recovery_manifest','feasibility_audit']:
 checks[k+'_hash']=hashlib.sha256((E/manifest[k]).read_bytes()).hexdigest()==manifest[k+'_sha256']
baseline=read('config/d10_baseline.json')
checks['protected_blobs_match_main_tree']=all(tree.get(k,{}).get('sha')==v for k,v in baseline['protected_blobs'].items())
checks['d9_d10_main_metrics_identical']=s['performance']==d['frozen_d9_v2_performance']
checks['ir_times_te']=math.isclose(p['information_ratio']*p['annualized_tracking_error'],p['annualized_excess_return'],abs_tol=1e-14)
c=d['attribution']['oos']['annualized_contributions']
checks['attribution_closes']=math.isclose(sum(c[k] for k in ['industry','style','specific_net','benchmark_bridge','unmodelled_bridge']),p['annualized_excess_return'],abs_tol=1e-12)
checks['specific_share']=math.isclose(c['specific_net']/p['annualized_excess_return'],d['attribution']['oos']['specific_share_net'],abs_tol=1e-12)
checks['july_difference']=math.isclose(d['july_2026']['portfolio_return']-d['july_2026']['benchmark_return'],d['july_2026']['excess_return'],abs_tol=1e-12)
checks['all_downloaded_sources_match_git_blob']=True
assert all(checks.values()),checks
(OUT/'D11_verification.json').write_text(json.dumps({'source_commit':SHA,'checks':checks,'source_files':len(hashrows),'numeric_index_rows':len(numrows),'reran_models':False,'reran_backtests':False,'full_data_available':False,'d10_diagnostic_acceptance_passed':False},ensure_ascii=False,indent=2)+'\n')

# ReportLab PDF from the same Markdown. Explicit section breaks preserve review structure.
pdfmetrics.registerFont(TTFont('D11ReportSans',str(ARCHIVE/'fonts/D11ReportSans.ttf')))
pdfmetrics.registerFontFamily('D11ReportSans',normal='D11ReportSans',bold='D11ReportSans',italic='D11ReportSans',boldItalic='D11ReportSans')
styles={
 'body':ParagraphStyle('Body',fontName='D11ReportSans',fontSize=10,leading=15.2,textColor=colors.HexColor('#243448'),spaceAfter=8,wordWrap='CJK'),
 'title':ParagraphStyle('Title',fontName='D11ReportSans',fontSize=23,leading=31,textColor=colors.HexColor('#143a48'),spaceAfter=18,wordWrap='CJK'),
 'h2':ParagraphStyle('H2',fontName='D11ReportSans',fontSize=17,leading=23,textColor=colors.HexColor('#17675c'),spaceAfter=14,wordWrap='CJK',keepWithNext=True),
 'cell':ParagraphStyle('Cell',fontName='D11ReportSans',fontSize=8.3,leading=11.9,textColor=colors.HexColor('#243448'),wordWrap='CJK'),
 'head':ParagraphStyle('Head',fontName='D11ReportSans',fontSize=8.5,leading=12.5,textColor=colors.white,wordWrap='CJK'),
 'small':ParagraphStyle('Small',fontName='D11ReportSans',fontSize=8.2,leading=12,textColor=colors.HexColor('#536477'),spaceAfter=7,wordWrap='CJK'),
}
def inline(t):
 t=html.escape(t).replace('w₀','w0').replace('Σ','Sigma').replace('μ','mu').replace('σ','sigma').replace('√','sqrt ')
 t=re.sub(r'\[([^\]]+)\]\(([^\)]+)\)',lambda m:f'<link href="{m.group(2)}" color="#17675c">{m.group(1)}</link>',t)
 t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
 t=re.sub(r'`([^`]+)`',r'<font color="#17675c">\1</font>',t)
 return t
story=[];lines=content.splitlines();i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line=='<!-- PAGEBREAK -->':story.append(PageBreak());i+=1;continue
 if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['title']));i+=1;continue
 if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[-: ]+',x) for x in row):block.append(row)
   i+=1
  n=len(block[0]);width=499
  if n==2:ratios=[.26,.74]
  elif n==3:ratios=[.24,.36,.40] if len(block)>8 else [.36,.32,.32]
  elif n==4:ratios=[.19,.29,.29,.23]
  elif n==5:ratios=[.10,.36,.18,.18,.18] if block[0][0]=='层' else [.36,.16,.16,.16,.16]
  else:ratios=[.20]+[.80/(n-1)]*(n-1)
  cells=[[Paragraph(inline(v),styles['head' if r==0 else 'cell']) for v in row] for r,row in enumerate(block)]
  tb=Table(cells,colWidths=[width*r for r in ratios],repeatRows=1,hAlign='LEFT')
  tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1d5360')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#17675c'))]))
  story.extend([tb,Spacer(1,11)]);continue
 m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
 if m:
  from PIL import Image as PILImage
  imagefile=OUT/m.group(2);iw,ih=PILImage.open(imagefile).size
  ww=430 if 'd10_oos' in str(imagefile) else 499
  story.extend([Image(str(imagefile),width=ww,height=ww*ih/iw),Spacer(1,4),Paragraph(inline(m.group(1)),styles['small'])]);i+=1;continue
 small=line.startswith('来源：') or line.startswith('检索日：') or line.startswith('- [W')
 story.append(Paragraph(inline(line),styles['small' if small else 'body']));i+=1
def page(canvas,doc):
 w,h=A4;canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d7e2e4'));canvas.line(48,h-37,w-48,h-37)
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526979'));canvas.drawString(48,h-28,'CSI500 RESEARCH / D11 / 2026-09-05')
 canvas.drawRightString(w-48,h-28,'D9-V2: No-Go | main 42c8870')
 canvas.line(48,36,w-48,36);canvas.drawString(48,24,'Research report - post-hoc amendment and known limitations disclosed');canvas.drawRightString(w-48,24,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(OUT/'D11研究报告.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=54,bottomMargin=49,title='中证500指数增强原型 D11研究报告',author='庭頫 · 研究报告整理')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(json.dumps({'output':str(OUT),'checks':checks,'sources':len(hashrows),'index_rows':len(numrows)},ensure_ascii=False))

shutil.copy2(ARCHIVE/'verify_d11.py',OUT/'verify_d11.py')
(OUT/'README.md').write_text('Rebuilt archival D11 report. Run python verify_d11.py here. For current repair status see D11运行说明.md and reports/D11修复核查报告.md in the repository. No repaired historical performance is generated by this script.\n')
