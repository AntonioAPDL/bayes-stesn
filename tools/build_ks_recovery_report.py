#!/usr/bin/env python3
"""Render surviving KS evidence without manufacturing the missing final run directory.

Uses a hash-verified published main audit and explicitly labeled transcript recovery.
No dataset/model/fit/forecast import; no original full-report gate is bypassed.
"""
from __future__ import annotations
import argparse, base64, hashlib, html, importlib.util, io, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from pypdf import PdfReader

AUDIT_SHA='fe4448818eda7e7f3ead715d989e5900db5464d47ddd446ce357c161e792f4c0'
ORDER=['bayes_stesn','separable_gp','basis_gru','ridge_esn','pod_nvar','linear_var','periodic_cnn','persistence','training_mean']
NATIVE=ORDER[:3]
LABEL={'bayes_stesn':'Bayesian ST-ESN','separable_gp':'Separable GP','basis_gru':'Basis GRU','ridge_esn':'Ridge ESN','pod_nvar':'POD-NVAR','linear_var':'Linear/POD VAR','periodic_cnn':'Periodic CNN','persistence':'Persistence','training_mean':'Training mean'}
SHORT={'bayes_stesn':'Bayes','separable_gp':'GP','basis_gru':'GRU','ridge_esn':'ESN','pod_nvar':'NVAR','linear_var':'VAR','periodic_cnn':'CNN','persistence':'Persist.','training_mean':'Mean'}
COL=dict(zip(ORDER,['#b23b41','#3b69a0','#178c89','#9a6a31','#858585','#7256a7','#31824a','#c78b19','#293d4d']))
LEADS=[1,5,10,25,50,100]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())
def atomic_bytes(path,payload):
    path=Path(path);tmp=path.with_name('.'+path.name+'.recovery-writing')
    with tmp.open('wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    if path.stat().st_size!=len(payload) or sha(path)!=hashlib.sha256(payload).hexdigest():
        raise IOError('Artifact bytes changed after atomic publication: '+str(path))
def num(x,d=4):
    if x is None:return 'Unavailable'
    x=float(x)
    if not math.isfinite(x):raise ValueError('Nonfinite values must remain failures')
    return f'{x:.3e}' if x and (abs(x)>=1e5 or abs(x)<1e-4) else f'{x:.{d}f}'
def est(r,k,d=3):
    v=r.get(k,{})
    if not v or v.get('mean') is None:return 'Unavailable'
    a=num(v['mean'],d);ci=v.get('ci95')
    return a+('\n['+num(ci[0],d)+', '+num(ci[1],d)+']' if ci else '')
def plain(t):return str(t).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')

class Evidence:
    def __init__(self,repo,recovery,out):
        self.repo,self.recovery,self.out=map(Path,(repo,recovery,out));self.out.mkdir(parents=True,exist_ok=True)
        self.inputs=[];p=self.repo/'docs/applications/evidence/ks_main_tracks_audit_v2.json'
        if sha(p)!=AUDIT_SHA:raise ValueError('Published main audit hash does not match the retained verified artifact')
        self.audit=self.load(p,'E1: published, hash-verified independent audit')
        if self.audit['passed'] is not True or self.audit['discrepancies']:raise ValueError('Main audit did not pass')
        self.frozen=self.load(self.repo/'docs/applications/evidence/ks_frozen_specification.json','E1: published pretest specification')
        self.pre=self.load(self.repo/'docs/applications/evidence/ks_preflight.json','E1: published preflight evidence')
        self.cfg=self.load(self.repo/'configs/applications/ks_experiment.json','E1: published initial candidate/budget specification')
        self.trans=self.load(self.recovery,'E2: new partial transcription; original extension audit missing')
        q=self.recovery.parent/'retained_contiguous_diagnosis_facts.json';self.geometry=self.load(q,'E2: partial diagnostic transcript') if q.exists() else None
        if self.audit['freeze_identity']!=self.frozen['freeze_identity'] or self.trans['freeze_identity']!=self.frozen['freeze_identity']:raise ValueError('Mixed frozen identities')
        self.rows=self.audit['rows'];self.leads=self.audit['predeclared_horizons'];self.pairs=self.audit['paired_comparisons']
        if len(self.rows)!=17 or len(self.leads)!=102:raise ValueError('Incomplete published main audit population')
        if len({(r['track'],r['model']) for r in self.rows})!=17:raise ValueError('Duplicate main cohort')
        for r in self.rows:
            if r['selected_config']!=self.frozen['selected'][r['track']][r['model']]['id']:raise ValueError('Selected configuration differs')
            c=r['counts']
            if c['scheduled_origins']!=400 or c['finite_successful_origins']+c['failed_origins']+c['pending_origins']!=400:raise ValueError('Bad full schedule denominator')
            if r['rmse']['n']!=c['complete_trajectories']:raise ValueError('Accuracy population mismatch')
        for t in ['A','B']:
            for r in self.track(t):
                if sorted(x['horizon'] for x in self.leads if x['track']==t and x['model']==r['model'])!=LEADS:raise ValueError('Incomplete six-lead record')
        oldspec=importlib.util.spec_from_file_location('ks_old_report',self.repo/'tools/build_ks_pdf.py');mod=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(mod)
        self.references=mod.REFERENCES
        custody=self.recovery.parent/'recovery_custody.json'
        self.custody=self.load(custody,'E1 recovery custody check of published main extract; not a rerun of lost receipts') if custody.exists() else None
        self.example=None;self.example_receipt=None
        example_dir=self.out/'recovered_example'
        if (example_dir/'artifact_receipt.json').exists():
            self.example_receipt=self.load(example_dir/'artifact_receipt.json','E3: newly recomputed fixed example, verified native fit/runtime')
            self.example=self.load(example_dir/'reconstruction.json','E3: new forecast receipt, no refit or reselection')
            ex=self.example;receipt=self.example_receipt;case=ex['binding']['case']
            if (receipt['status']!='complete_visual_review_passed' or receipt['freeze_identity']!=self.frozen['freeze_identity'] or
                    ex['binding']['freeze_identity']!=self.frozen['freeze_identity'] or not all(ex['binding']['checks'].values()) or
                    (case['track'],case['model'],case['trajectory'],case['first_target'],case['paths'])!=('A','bayes_stesn',0,8000,512) or
                    ex['new_fit_count']!=0 or ex['new_selection_count']!=0 or ex['new_forecast_origin_count']!=1):
                raise ValueError('Recomputed example is not the verified predeclared case')
            for item in receipt['files']:
                if item['name'].endswith(('.png','.pdf','.mp4')) or item['name']=='reconstruction.json':
                    q=example_dir/item['name']
                    if q.stat().st_size!=item['bytes'] or sha(q)!=item['sha256']:raise ValueError('Recomputed example artifact changed')
        self.figures=[]
    def load(self,p,level):
        self.inputs.append({'path':str(p.relative_to(self.repo)) if p.is_relative_to(self.repo) else (str(p.relative_to(self.out)) if p.is_relative_to(self.out) else p.name),'bytes':p.stat().st_size,'sha256':sha(p),'evidence_level':level})
        return read(p)
    def track(self,t):return sorted([r for r in self.rows if r['track']==t],key=lambda r:ORDER.index(r['model']))
    def lead(self,t,m):return sorted([r for r in self.leads if r['track']==t and r['model']==m],key=lambda r:r['horizon'])
    def figsave(self,fig,name,caption):
        directory=self.out/'figures';directory.mkdir(exist_ok=True)
        for ext in ['png','pdf']:
            memory=io.BytesIO();fig.savefig(memory,format=ext,dpi=180,bbox_inches='tight',facecolor='white')
            atomic_bytes(directory/(name+'.'+ext),memory.getvalue())
        plt.close(fig);self.figures.append({'name':name,'caption':caption,'png':directory/(name+'.png'),'pdf':directory/(name+'.pdf')})
    def figures_build(self):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.constrained_layout.use':True,'legend.frameon':False})
        fig,axes=plt.subplots(2,2,figsize=(10.5,7.1))
        for row,t in enumerate(['A','B']):
            for col in [0,1]:
                ax=axes[row,col]
                for r in self.track(t):
                    m=r['model']
                    if col==1 and m=='pod_nvar':continue
                    z=self.lead(t,m);y=np.array([q['rmse']['mean'] for q in z]);ci=np.array([q['rmse']['ci95'] for q in z])
                    ax.plot(LEADS,y,'o-',ms=3,lw=1.5,color=COL[m],label=SHORT[m]);ax.fill_between(LEADS,ci[:,0],ci[:,1],color=COL[m],alpha=.10)
                if col==0:ax.set_yscale('log')
                ax.set_title(f'Track {t}: '+('all methods, logarithmic scale' if col==0 else 'readable companion, NVAR omitted'))
                ax.set_xlabel('Forecast lead (saved steps; 0.25 time units each)');ax.set_ylabel('Episode-balanced RMSE');ax.set_xticks([1,10,25,50,100])
        handles,labels=axes[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=9)
        self.figsave(fig,'audited_error_leads','E1. Only the six audited leads are plotted; connecting segments are visual guides, not recovered 100-lead curves. Shading is the recorded pointwise trajectory-bootstrap 95% interval. Full-range panels retain NVAR extremes; readable companions omit only NVAR, explicitly. Track A NVAR uses 17 complete trajectories and has undefined full-schedule RMSE.')
        fig,axes=plt.subplots(3,2,figsize=(10.5,8.5))
        for col,t in enumerate(['A','B']):
            for row,(key,title) in enumerate([('crps','CRPS'),('coverage_0.9','90% coverage'),('width_0.9','90% interval width')]):
                ax=axes[row,col]
                for m in NATIVE:
                    z=self.lead(t,m);y=np.array([q[key]['mean'] for q in z]);ci=np.array([q[key]['ci95'] for q in z]);ax.plot(LEADS,y,'o-',ms=3,color=COL[m],label=SHORT[m]);ax.fill_between(LEADS,ci[:,0],ci[:,1],color=COL[m],alpha=.10)
                if key=='coverage_0.9':ax.axhline(.9,color='black',ls='--',lw=.8);ax.set_ylim(0,1.02)
                ax.set_title(f'Track {t}: {title}');ax.set_xlabel('Lead / saved steps');ax.set_xticks([1,10,25,50,100])
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        self.figsave(fig,'audited_probability_leads','E1. Qualified native predictive distributions, 20 trajectories each. Six audited leads and pointwise trajectory-bootstrap intervals. Coverage concerns the clean field. Early undercoverage and late overcoverage cannot be inferred away by a favorable aggregate.')
        fig,axes=plt.subplots(2,2,figsize=(10,6.5))
        levels=[.5,.8,.9,.95]
        for col,t in enumerate(['A','B']):
            for m in NATIVE:
                r=next(x for x in self.track(t) if x['model']==m)
                for row,prefix in [(0,'coverage'),(1,'width')]:
                    axes[row,col].plot(levels,[r[f'{prefix}_{q}']['mean'] for q in levels],'o-',color=COL[m],label=SHORT[m])
            axes[0,col].plot([.45,1],[.45,1],'--',color='black',lw=.8);axes[0,col].set_title('Track '+t+': marginal calibration');axes[0,col].set_ylabel('Observed coverage');axes[0,col].set_ylim(.25,1.02)
            axes[1,col].set_title('Track '+t+': interval width');axes[1,col].set_ylabel('Physical-unit width')
            for ax in axes[:,col]:ax.set_xlabel('Nominal central probability');ax.set_xticks(levels)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        self.figsave(fig,'audited_calibration_width','E1. Ordinary 100-step aggregates of central 50%, 80%, 90% and 95% intervals. Exact confidence intervals are given in the probability tables. The diagonal is nominal coverage; wider intervals alone do not establish better forecasting.')
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
        for r in self.rows:
            c=r['resource_reconstruction'];x=c['fit_seconds']+c['forecast_seconds'];y=r['rmse']['mean'];m=r['model'];marker='o' if r['track']=='A' else '^'
            for j,ax in enumerate(axes):
                if j==1 and m=='pod_nvar':continue
                ax.scatter(x,y,marker=marker,s=48,color=COL[m],edgecolor='white',linewidth=.5)
                ci=r['rmse']['ci95'];ax.vlines(x,ci[0],ci[1],color=COL[m],alpha=.7,lw=1)
        for ax in axes:ax.set_xscale('log');ax.set_xlabel('Recorded fit + forecast seconds');ax.set_ylabel('Mean trajectory RMSE')
        axes[0].set_yscale('log');axes[0].set_title('Full range, all methods');axes[1].set_title('Readable companion: NVAR omitted')
        from matplotlib.lines import Line2D
        handles=[Line2D([0],[0],color=COL[m],marker='o',lw=0,label=SHORT[m]) for m in ORDER]
        fig.legend(handles=handles,loc='outside lower center',ncol=9)
        self.figsave(fig,'audited_accuracy_compute','E1. Circles: A; triangles: B. Cost sums the surviving audit\'s selected-fit receipt times and recorded forecast times; it is not cumulative screening/campaign cost. Full-range plot includes enormous finite A NVAR error. Companion omits NVAR in both tracks. Main accuracy populations remain as labeled in the tables.')
        seed=self.trans['matched_seed_displayed_results'];fig,axes=plt.subplots(1,2,figsize=(10,4.1));fig.suptitle('E2 | Transcript-recovered seed summaries; original episode artifacts unavailable',fontsize=10)
        for j,t in enumerate(['A','B']):
            for suffix,m in [('bayes','bayes_stesn'),('gru','basis_gru'),('esn','ridge_esn'),('cnn','periodic_cnn')]:
                key=t+'_'+suffix
                if key in seed['rmse']:axes[j].plot([0,1,2],seed['rmse'][key],'o-',color=COL[m],label=SHORT[m])
            axes[j].set_xticks([0,1,2],['20261105','20261106','20261107']);axes[j].set_title('Track '+t+': same four trajectories');axes[j].set_ylabel('Mean trajectory RMSE');axes[j].legend()
        self.figsave(fig,'transcript_seed_sensitivity','E2. Transcript-recovered descriptive means on exactly trajectories [0,5,10,15], including the main seed. These are not the 20-trajectory main means, additional scientific replicates, or posterior draws. Original matched episode records are unavailable; no new interval is inferred from rounded means.')
        c=self.trans['C_displayed_results'];fig,axes=plt.subplots(1,2,figsize=(10.5,4.8));fig.suptitle('E2 | Transcript-recovered contiguous outcomes; original full audit unavailable',fontsize=10);names=[SHORT[r['model']] for r in c];x=np.arange(len(c))
        vals=[float(r['rmse']) if r['rmse'] is not None else np.nan for r in c];axes[0].bar(x,vals,color=[COL[r['model']] for r in c]);axes[0].set_yscale('log');axes[0].set_xticks(x,names,rotation=35,ha='right');axes[0].set_ylabel('Mean RMSE (log scale)');axes[0].set_title('C accuracy: NVAR undefined');axes[0].text(7,.6,'undefined',rotation=90,ha='center',fontsize=8)
        fail=np.array([r['nonfinite_failed'] for r in c]);div=np.array([r['finite_divergent'] for r in c]);axes[1].bar(x,div/400,color='#bd843d',label='Finite divergence');axes[1].bar(x,fail/400,bottom=div/400,color='#ad3740',label='Nonfinite forecast failure');axes[1].set_xticks(x,names,rotation=35,ha='right');axes[1].set_ylim(0,1.03);axes[1].set_ylabel('Fraction of all 400 origins');axes[1].legend(loc='upper right',fontsize=8);axes[1].set_title('C reliability: full scheduled denominator')
        self.figsave(fig,'transcript_contiguous_outcomes','E2. Retained contiguous-track summaries, not a reconstructed full audit. All Bayesian origins are finite but exceed the predeclared divergence threshold. NVAR has 260 nonfinite failures plus 37 finite divergences; no complete trajectory and no aggregate RMSE. No raw rollout or per-trajectory record is recreated.')
        if self.example:
            d=self.out/'recovered_example';self.figures.append({'name':'recomputed_example','caption':'E3. Newly recomputed Bayesian A forecast from the retained original fit, trajectory 0 / first target 8000. Only the 12 official withheld target rows are plotted; the 39 input sites are shown separately as a coordinate rug. No whole-field score is implied. The source receipt verifies 512 paths, the frozen seed and 24 bindings; original full-study output recovery is not claimed.','png':d/'recomputed_A_bayes_t00_a8000_heatmap.png','pdf':d/'recomputed_A_bayes_t00_a8000_heatmap.pdf'})
            self.figures.append({'name':'recomputed_intervals','caption':'E3. Newly recomputed fixed-origin forecast: the three predeclared Track A target-array positions [0,5,11], observation-panel site IDs [1,30,60], physical coordinates [0.34375,10.3125,20.625]. Clean truth, complete-path predictive mean and central 50%/90% bands are shown against physical lead time. These sites were withheld from all model inputs. Bands summarize the same 512 complete predictive paths, not a Gaussian approximation fitted to deterministic controls.','png':d/'recomputed_A_bayes_t00_a8000_intervals.png','pdf':d/'recomputed_A_bayes_t00_a8000_intervals.pdf'})

class Report:
    def __init__(self,e):
        self.e=e;self.story=[];self.blocks=[];self.assets=[]
        fontdir=Path('/usr/share/fonts/truetype/dejavu')
        if not (fontdir/'DejaVuSans.ttf').exists():fontdir=Path('/opt/codex/runtimes/codex-primary-runtime/dependencies/fonts')
        candidates=list(fontdir.rglob('DejaVuSans.ttf'))
        if candidates:
            normal=candidates[0];bold=normal.with_name('DejaVuSans-Bold.ttf');pdfmetrics.registerFont(TTFont('KS',str(normal)));pdfmetrics.registerFont(TTFont('KSB',str(bold)));self.font='KS';self.bold='KSB'
        else:self.font='Helvetica';self.bold='Helvetica-Bold'
        self.styles={'body':ParagraphStyle('body',fontName=self.font,fontSize=9,leading=13,spaceAfter=8,textColor=colors.HexColor('#20313d')),'small':ParagraphStyle('small',fontName=self.font,fontSize=7.6,leading=10.2,spaceAfter=6,textColor=colors.HexColor('#4b5c66')),'h1':ParagraphStyle('h1',fontName=self.bold,fontSize=18,leading=23,spaceAfter=14,textColor=colors.HexColor('#17384b')),'h2':ParagraphStyle('h2',fontName=self.bold,fontSize=11.2,leading=15,spaceBefore=8,spaceAfter=7,textColor=colors.HexColor('#207776')),'cell':ParagraphStyle('cell',fontName=self.font,fontSize=7.5,leading=10),'head':ParagraphStyle('head',fontName=self.bold,fontSize=7.5,leading=10,textColor=colors.white)}
    def p(self,text,small=False):
        text=plain(text);self.story.append(Paragraph(html.escape(text).replace('\n','<br/>'),self.styles['small' if small else 'body']));self.blocks.append('<p'+(' class="small"' if small else '')+'>'+html.escape(text).replace('\n','<br>')+'</p>')
    def h(self,text,level=1,new=True):
        if level==1 and self.story and new:self.story.append(PageBreak())
        self.story.append(Paragraph(html.escape(text),self.styles['h1' if level==1 else 'h2']));self.blocks.append(f'<h{level}>'+html.escape(text)+f'</h{level}>')
    def table(self,headers,rows,widths=None):
        if not rows:self.p('No retained eligible values.');return
        w=515; widths=widths or [w/len(headers)]*len(headers)
        factor=w/sum(widths);widths=[x*factor for x in widths]
        def cell(x,head=False):return Paragraph(html.escape(plain(x)).replace('\n','<br/>'),self.styles['head' if head else 'cell'])
        data=[[cell(x,True) for x in headers]]+[[cell(x) for x in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#214b60')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f2f6f7'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.white)]));self.story.extend([t,Spacer(1,8)])
        self.blocks.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(plain(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(plain(x)).replace('\n','<br>')+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    def fig(self,name,maxheight=430):
        a=next(x for x in self.e.figures if x['name']==name)
        from PIL import Image as PIL
        raw=a['png'].read_bytes();buffer=io.BytesIO(raw)
        with PIL.open(buffer) as im:im.load();w,h=im.size
        buffer.seek(0);self.assets.append(buffer)
        scale=min(515/w,maxheight/h);self.story.append(Image(buffer,width=w*scale,height=h*scale));self.p(a['caption'],True)
        encoded=base64.b64encode(raw).decode();self.blocks.append('<figure><img alt="'+html.escape(a['caption'])+'" src="data:image/png;base64,'+encoded+'"><figcaption>'+html.escape(a['caption'])+'</figcaption></figure>')
    def eq(self,tex):
        from matplotlib.mathtext import math_to_image
        buf=io.BytesIO();math_to_image('$'+tex+'$',buf,dpi=190,format='png');buf.seek(0)
        from PIL import Image as PIL
        with PIL.open(buf) as im:w,h=im.size
        factor=min(72/190,515/w);self.assets.append(buf);self.story.append(Image(buf,width=w*factor,height=h*factor));self.story.append(Spacer(1,9));self.blocks.append('<figure class="equation"><img style="max-width:850px;width:auto;height:auto" alt="'+html.escape(tex)+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'"></figure>')
    def build_content(self):
        e=self.e;f=e.frozen
        self.h('Kuramoto-Sivashinsky forecasting\nAudited results and recovery report',new=False)
        self.p('Custom, literature-grounded Bayesian ST-ESN comparator experiment | 5 November 2026')
        self.h('Status: scientific execution completed; artifact handoff incomplete',2)
        self.p('The retained coordinator status records all 900 scheduled jobs terminal: 870 complete, 29 failed and one inference-unqualified; no pending jobs. Recorded scientific worker time was 6.033 hours within the original eight-hour ceiling. These are historical terminal-status facts [E2]. After execution and independent audits, the original transient workspace became unavailable before the full final result bundle and comprehensive report were durably published. A subsequent runtime notice states that automated workspace maintenance pruned earlier scratch/top-level directories to recover storage space. This report does not infer finer internal details of that maintenance.')
        self.p('This report salvages available evidence. It does not certify that missing raw forecasts, per-trajectory results, fitted models, complete posterior diagnostics, or the original final report have been recovered. It does not rerun the experiment or manufacture their replacements. The full application handoff requested by the user remains incomplete.')
        self.table(['Evidence level','What survives','What this permits'],[['E1: published audit','Exact hash-verified main A/B audit, frozen specification, code, registry, preflight and earlier checks','Report its 17 main aggregates, 102 prespecified lead records, intervals, paired comparisons, failures and recorded costs'],['E2: transcript recovery','New summaries transcribed from pre-outage tool output/messages for C, sensitivity, final audit and diagnosis','Clearly labeled historical findings at the precision retained; no claim of recovered original source receipts'],['E3: new fixed-case replay','One predeclared Bayesian A forecast recomputed from the retained fit with verified native identities','An actual heatmap and compact movie for that single case; not a recovered full-run archive'],['Unavailable','Original complete final run directory, raw path arrays, full final per-episode/lead files and selected checkpoint set','No new full-run verification, heatmaps, interval traces or animations fabricated from aggregates']],[95,205,215])
        self.h('Scientific findings',2)
        self.p('A: the selected Bayesian ST-ESN has higher RMSE than every other fully reliable family, and higher CRPS than GP and GRU. B: the small periodic CNN has the lowest mean RMSE; Bayesian ST-ESN remains worse than CNN, VAR, GRU and GP on paired RMSE. NVAR exhibits extreme rare errors, so low MAE or a long valid-prediction time does not establish reliability. C: retained results show catastrophic Bayesian spatial-transfer failure and 260/400 NVAR forecast failures. [E1/E2, separated in subsequent tables.]')
        self.p('Main audit SHA-256: '+AUDIT_SHA,True)
        self.h('1. Frozen scientific contract and scoring')
        self.eq(r'u_t=-u u_x-u_{xx}-u_{xxxx},\qquad x\in[0,22),\quad u(x+22,t)=u(x,t)')
        self.p('Prepared numerical data are unchanged: 20 core trajectories, 10,000 saved states each; three independent pilot trajectories, 2,000 states each. Solver truth has 128 sites; the standard observation panel has 64, every second truth-grid point. Saved cadence is 0.25 physical units; the 100-step horizon spans 25 units. No solver field, derivatives, known-PDE integrator or future labels are extra training inputs. [E1: preflight/protocol]')
        self.table(['Contract','Frozen value'],[['Chronological intervals','Training targets [0,6000); validation [6000,8000); test [8000,10000), zero-based half-open'],['Track A: strict spatial application','39 interleaved training sites, 13 validation sites, 12 test sites; training-site histories only'],['Track B: observed dynamics','All 64 panel sites for permitted training/history and future scoring; separately selected models'],['Track C: robustness','40/12/12 contiguous split; transfer frozen A configurations to newly fitted permitted training transforms and parameters'],['Forecast window','History y[a-128:a], targets y[a:a+100], adapter origin a-1. Test starts 8000,8100,...,9900'],['Predictive effort','512 complete paths per native origin, generated in batches of 64; static parameters fixed along each Bayesian path'],['Replicates','20 trajectories equally weighted. Twenty origins per trajectory remain dependent; no site/time/path pseudoreplication'],['Transform boundary','Delivered trajectory/protocol-specific scalar training-only transform; features, POD and covariance obey the same access boundary']],[125,390])
        self.p('Each origin is autonomous for its full horizon. Later origins may use newly available earlier observations at permitted sites, but there is no parameter refit or truth update inside the forecast interval. Recurrent/lag state resets between trajectories. The facade constructs evaluation labels internally; separation/invariance checks concern model-returned inputs, not a claim of future-free physical disk access.')
        self.p('Point scores use the complete-path predictive mean for native models. Within each trajectory, MSE is averaged across all 20 origins and scored cells before taking its square root; reported RMSE averages these trajectory RMSEs. MAE, CRPS and coverage/width are averaged within trajectory and then across trajectories. Reported intervals are 2,000-resample trajectory-cluster percentile intervals, conditional on selected configurations. A nonfinite origin excludes its entire trajectory from accuracy summaries but remains in the full-schedule failure denominator.')
        self.p('VPT is the first lead where the primary point forecast\'s spatial RMSE/training SD exceeds 0.4, right-censored at 100. It is not a distribution of pathwise crossing times. Finite divergence flags any lead exceeding normalized RMSE 10; nonfinite rollouts are separate failures. No clipping, replacement seeds or favorable-origin substitution is allowed. Physical time is 0.25 times the reported step count.')
        self.h('2. The preserved Bayesian spatial/ST-ESN')
        self.eq(r'y_t=\Phi C z_t+e_t,\quad e_t\sim N_S(0,\sigma^2 R),\quad R=I_S+qK_\ell')
        self.eq(r'\mathrm{vec}(C)\mid\sigma^2,\lambda\sim N_{pd}(0,\sigma^2\lambda^{-1}I),\quad\sigma^2\sim\mathrm{IG}(2,0.05)')
        self.eq(r'\log\lambda\sim N(\log 0.1,1.5^2),\quad\log q\sim N(0,1.25^2)')
        self.p('The accepted Gaussian likelihood, scalar ridge readout hierarchy, relative spatial covariance and inverse-gamma variance prior are retained. All readout coefficients, including intercepts, receive this ridge prior. The IG convention is shape/scale, with density proportional to x^(-a-1) exp(-b/x). There is no horseshoe, quantile likelihood, deterministic posterior replacement or relabeled generic ESN. Conditional posterior inference is given the validation-selected architecture and hyperparameters, not averaging over the selection process. [E1: code/registry/freeze]')
        self.table(['Hyperprior / representation','Reference bayes_00','A bayes_03','B bayes_01'],[['Mean log spatial range','log(0.18)','log(0.18)','log(0.30)'],['SD log spatial range','0.65','0.65','0.65'],['Spatial columns p','12','32','20'],['Temporal columns d','65','225','113'],['Readout coefficients p x d','780','7,200','2,260'],['Conditioning / likelihood targets','80 / 5,920','1,024 / 4,976','512 / 5,488'],['PCA modes / units / squares','8 / 32 / 16','16 / 128 / 64','8 / 64 / 32']],[155,120,120,120])
        self.p('K is the positive-definite Matérn-3/2 covariance restricted to the unit-circle embedding e(x)=(cos(2*pi*x/22), sin(2*pi*x/22)). Range ell uses embedding chord units. Reference prior constants and KS-selected constants are distinguished: B changes the log-range mean within the same prior family. Spatial centers are predetermined on the circle; basis column statistics use permitted static training coordinates, never hidden response histories.')
        self.eq(r'\phi(x)=\left[1,\frac{r_j(x)-\bar r_{j,T}}{\max(s_{j,T},10^{-6})\sqrt{p-1}}\right]_{j=1}^{p-1}')
        self.p('For A/C, raw r contains two circle-coordinate columns plus 29 fixed-center radial columns. This geometry normalization is distinct from the delivered response transform and temporal-feature scaling. A different permitted geometry can induce very different target prior loading even under the same coefficient-prior family.')
        self.h('3. Reservoir, posterior engine and forecast law')
        self.eq(r'r_t=(I-D)r_{t-1}+D\tanh(Wr_{t-1}+U v_t+b)')
        self.p('The retained feature construction uses conditioning-prefix PCA scores at lags 1 and 2, fixed sparse leaky tanh dynamics, selected state squares and an intercept. Raw PCA scores drive the reservoir; readout score columns use 3*tanh(a/3). Recurrent scaling uses operator norm 0.9, heterogeneous leak pattern (0.2,0.5,0.9), connectivity 0.08 and input scale 0.5. This is distinct from the conventional ESN\'s scalar leak and spectral-radius scaling. No compressed irregular-site FFT is used.')
        self.eq(r'B=\Phi^T R^{-1}\Phi,\quad G=Z^T Z,\quad A=\lambda I+G\otimes B,\quad h=\mathrm{vec}(\Phi^T R^{-1}YZ)')
        self.eq(r'\mathrm{vec}(C)\mid\sigma^2,Y,\lambda,q,\ell\sim N(A^{-1}h,\sigma^2 A^{-1})')
        self.eq(r'Q=\mathrm{tr}(Y^T R^{-1}Y)-h^T A^{-1}h,\quad\sigma^2\mid Y,\lambda,q,\ell\sim\mathrm{IG}(a_0+ST/2,b_0+Q/2)')
        self.p('The Gaussian engine integrates C and sigma squared for collapsed hyperparameter inference, using spatial/temporal eigensystems and sufficient statistics rather than a full tensor-product design or dense coefficient covariance. Adaptive random-walk Metropolis operates on log(kappa)=log(lambda)+log(1+q), a unit-Jacobian change among log coordinates. Adaptation ends at warmup; this is not HMC. [E1: native audit/code]')
        self.p('Four sequential chains use 1,000 warmup and 2,000 retained iterations, with one predeclared 2,000-draw continuation. The frozen gate requires rank-normalized split/folded R-hat strictly below 1.01, bulk/tail ESS at least 400 and finite checks [7]. All hyperparameters are monitored; all readout coefficients and ten relevant functionals use at most 512 evenly spaced retained states per chain under a 256 MiB diagnostic cap. The published main audit records all main Bayesian fits as qualified; missing full final diagnostic arrays are not re-certified here.')
        self.p('A complete path selects one retained hyperstate and one joint readout/variance draw, keeps these static for 100 steps, draws the fitted spatial innovation field and updates its own permitted-site reservoir inputs. Innovations are iid in time conditional on this model, with preserved spatial covariance. Their role is model discrepancy for the clean simulated field; no arbitrary sensor-noise floor is added or removed.')
        self.p('The empirical CRPS is E|X-y| - 0.5 E|X-X\'| for the empirical predictive distribution. No iid unbiased finite-ensemble correction is claimed for correlated MCMC-derived paths. Validation predictive-simulation precision selected a common 512 paths before test access. That conditional simulation check is separate from MCMC diagnostics and between-trajectory uncertainty.')
        self.h('4. Comparator registry and declared adaptations')
        self.table(['Family','Mechanism and adaptation','Strict-site rule'],[['Separable GP','Matérn spatial covariance and temporal smoothness 1/2 or 3/2. Spatial eigendecomposition with exact finite-state Kalman likelihood/simulation [10]. Bounded likelihood optimization; conditional-on-estimated-hyperparameter uncertainty.','Joint spatial conditioning, preserving temporal covariance of the unseen remainder'],['Basis GRU','Training-coordinate orthonormal basis; GRU coefficient predictor + linear head, staged MSE/AdamW then spatial Gaussian residual likelihood. PyTorch reset-gate order differs from original Cho unit [11,12].','Same coordinate basis; joint fitted/target innovations; generated fitted-site state projected into next context'],['Linear/POD VAR','Regularized lagged state regression; optional training-only POD; mean squared loss + ridge.','Evolve native/POD state; output-only periodic linear interpolation'],['Sparse ridge ESN','Sparse tanh units, scalar leak, spectral radius, explicit intercept and [r,r squared] readout, state target [1,2]. 512 units -> 1,025 columns.','Same periodic decoder; not a matched deterministic ablation of Bayesian architecture'],['POD-NVAR','Quadratic polynomial in delayed training-only POD scores, increment target [3]. 1+d+d(d+1)/2 features; selected ranks/taps yield 153 or 561.','Latent recurrence; decoder applied only for target output'],['Periodic CNN','Small circular residual convolutional increment emulator, width 32, three blocks, kernel 5, 20 selected epochs. CPU-feasible PyTorch implementation [5].','B only; no zero-filled hidden-site adaptation'],['Mean / persistence','Per-observed-site training mean or last permitted observation. Point-only anchors.','Parameter-free periodic piecewise-linear decoder']],[100,270,145])
        self.p('The ridge objective is N^(-1)||XW-Y||^2 + lambda sum of squared non-intercept weights. Numerical lambda is not equated with a summed-loss penalty or Bayesian prior variance. All conventional vector baselines keep their own observed-site or POD recurrence; decoding never feeds hidden truth or silently smooths/reprojects their state each step.')
        self.p('The GRU internally uses a training-only chronological stopping tail and refits the chosen epoch on the full permitted panel; this is not a new hyperparameter configuration tuned on each trajectory\'s official validation. CNN uses a common validation-selected fixed epoch budget. GP/GRU residual-optimizer qualification accepts finite, non-worsening successful bounded optimization; convergence by relative objective tolerance need not imply projected gradient below gtol. GRU residual optimization does not certify neural-stationarity convergence.')
        self.p('These are custom KS adaptations. Pathak/Vlachas motivate reservoir comparison, Gauthier motivates delay-polynomial features, and FNO/APEBench provide neural-emulator context. A circular CNN was implemented after a measured CPU pilot; no FNO result, published leaderboard reproduction or additional benchmark dataset is implied.')
        self.h('5. Selection opportunity and frozen winners')
        self.p('Both tracks completed selection before the test gate. Initial screening used core trajectories [0,5], validation starts [6000,6500,7000,7500], full permitted training data and one fixed model seed. Two eligible candidates per family were promoted to [0,5,10,15,4,9] at eight declared validation starts. Anchors remain in final tables. Near ties within 1% choose lower declared complexity, with one bounded validation-only boundary refinement. [E1: configuration/freeze]')
        self.table(['Family','Initial candidates / track','Declared search summary'],[['Bayesian ST-ESN','4','Reference plus 20/32 spatial columns; 32/64/128 units; 8/16 PCA modes; fixed accepted prior families'],['GP / GRU','2 / 4','GP temporal smoothness .5/1.5; GRU ranks 8/16/24, widths 32/64, lags 8/16, lr 1e-3/3e-4'],['VAR / ESN / NVAR','8 each','VAR ranks 8/16/32 and 1/2/4 lags; ESN 256/512 units; NVAR ranks 8/16, one/two taps, spacing 1/2; broad declared ridge proposals'],['CNN / anchors','4 / 1 each','Widths 16/32, blocks 2/3, epochs 20/60; anchors have no tuned parameter']],[110,110,295])
        self.p('Native probabilistic families optimize episode-balanced validation CRPS. Point-only families optimize episode-balanced squared error. Both give equal weight to mean losses in lead bands 1-10,11-25,26-50,51-100; these are different selection objectives and their raw values are not compared across families. Staged screening is an adaptation inspired by efficient allocation [6], not all Hyperband guarantees. The search budget was 2.5 worker hours out of eight; most work was reserved for selected fits, diagnostics and reporting.')
        selected=[]
        for t in ['A','B']:
            for r in e.track(t):
                v=f['selected'][t][r['model']];selected.append([t,LABEL[r['model']],v['id'],num(v['selection_loss'],5),'CRPS' if r['model'] in NATIVE else 'Squared error'])
        self.table(['Track','Family','Selected ID','Validation band loss','Objective'],selected,[40,137,110,118,110])
        self.h('5 continued. Selection caveat and effective settings')
        self.p('B NVAR instability was already visible before testing: nvar_03 promotion loss was 625,780,762.9157296. It was the only fully finite promoted/refined candidate; nvar_06 and a lower-ridge refinement each had a nonfinite validation rollout. Six other initial candidates were not revisited at promotion fidelity. Finite eligibility therefore did not establish stable forecasting or universal NVAR optimality. No post-test repair or reselection is made. [E1: published selection audit]')
        self.h('Selected effective settings',2)
        self.table(['Family / track','Frozen predictive configuration'],[
            ['Bayesian A / B','A: 32 spatial columns, 225 temporal columns, 128 units, 16 PCA modes; conditioning 1,024. B: 20 spatial, 113 temporal, 64 units, eight modes; conditioning 512. Same ridge/IG/lognormal hierarchy; range prior median .18 / .30.'],
            ['GP, both','Temporal nu .5; spatial nu 1.5. Initial spatial length .2 and temporal length 4, subsequently likelihood fitted; 60 iterations with a bounded 120-iteration continuation. ftol 1e-8; gtol 1e-5.'],
            ['GRU, both','Basis rank 16, width .25; hidden 64, lags 16; AdamW lr .001, weight decay 1e-4, batch 64, maximum 80 epochs. Training-tail fraction .2, patience 12, refit enabled. Residual covariance likelihood: 40 + at most 80 iterations, ftol 1e-9, gtol 1e-5.'],
            ['VAR, both','POD rank 16; four lags, unit spacing, state target; mean-loss ridge 1e-5; chunk 512.'],
            ['Ridge ESN, A / B','Both 512 units, connectivity .02, washout 128, state target, intercept + states + squares. A: radius .9, leak .5, input scale .2, ridge 1e-6. B: radius .7, leak .2, input scale 1, refined ridge .1.'],
            ['NVAR, A / B','Both POD rank 16, quadratic degree two, unit delay spacing, increment target. A: one tap, 153 features, ridge .001. B: two taps, 561 features, ridge 1e-6.'],
            ['CNN, B','Width 32, three circular residual blocks, kernel 5; 20 fixed epochs, batch 128, lr .001, weight decay 1e-6; increment target.'],
            ['Anchors','No tuning; per-observed-site mean or last value, then periodic interpolation for strict targets.']
        ],[105,410])
        self.p('The complete effective dictionaries, including fixed numerical guards and original named seed identities, remain in the published frozen specification and initial proposal file. Values above describe actual selected configurations; unused configuration aliases are not treated as distinct models.',True)
        for t in ['A','B']:
            rows=e.track(t);self.h('6'+t+'. Main point accuracy and full-schedule reliability')
            self.p('E1: exact published main-audit values. All models have 20 scheduled trajectories and 400 origins. Brackets are recorded 95% trajectory-cluster intervals. Accuracy is conditional on complete trajectories; numerical failures and finite divergence remain visible below.')
            self.table(['Family (complete n)','RMSE [95% CI]','MAE [95% CI]','NRMSE [95% CI]','VPT steps [95% CI]'],[[LABEL[r['model']]+'\nn='+str(r['counts']['complete_trajectories']),est(r,'rmse'),est(r,'mae'),est(r,'nrmse'),est(r,'vpt_steps',2)] for r in rows],[112,105,105,105,105])
            self.table(['Family','Finite /400','Nonfinite /400','Finite divergent /400','Full-schedule RMSE'],[[SHORT[r['model']],r['counts']['finite_successful_origins'],r['counts']['failed_origins'],r['counts']['finite_divergent_origins'],'Defined' if r['full_schedule_point_score_defined'] else 'Undefined'] for r in rows],[115,90,90,110,110])
            if t=='A':self.p('A NVAR has four failed origins in three trajectories; only 17 trajectories (340 origins) enter its accuracy population. Fifty-six additional finite origins in incomplete trajectories remain outside that accuracy population but inside reliability. Extremely large finite errors in complete trajectories remain in the 8.439e41 RMSE; no clipping or favorable-case replacement occurred.')
            else:self.p('All B trajectories are complete, but NVAR has two finite-divergent origins. Its RMSE retains those errors (maximum recorded origin RMSE 375.129376); the low MAE and high VPT describe typical behavior and do not erase rare instability. No observed failures for a method does not establish zero failure risk on new trajectories.')
            self.h('7'+t+'. Main probabilistic scores and interval behavior')
            nr=[r for r in rows if r['model'] in NATIVE]
            self.p('E1. All three native families have 20 qualified complete main trajectories. Predictive target: the clean field. Native point scores average the same 512 complete paths as the probabilistic scores; point-only competitors have no invented intervals.')
            self.table(['Family','CRPS [95% CI]','Spatial energy [95% CI]','Variogram [95% CI]'],[[LABEL[r['model']],est(r,'crps'),est(r,'spatial_energy'),est(r,'spatial_variogram')] for r in nr],[130,128,128,129])
            self.table(['Family','Central interval','Coverage [95% CI]','Physical width [95% CI]'],[[SHORT[r['model']],str(int(100*q))+'%',est(r,'coverage_'+str(q),4),est(r,'width_'+str(q),3)] for r in nr for q in [.5,.8,.9,.95]],[90,80,160,185])
            self.p('Spatial energy uses the Euclidean field norm divided by sqrt(number of target sites), with 1,024 seeded with-replacement empirical path pairs. Variogram uses power 0.5 and up to 128 geometry-only pairs with normalized inverse-chord-distance weights [9]. These are per-lead spatial scores, not a joint whole-temporal-path score. The pair approximation can make an empirical energy estimate slightly negative; no artificial nonnegative floor is imposed.',True)
        self.h('8. Paired scientific comparisons')
        self.p('E1. Differences are Bayesian minus comparator; negative favors Bayesian. Pairing retains exact common trajectory identities and origins. Confidence intervals resample trajectories, conditional on selection. A CI including zero does not establish equivalence. A NVAR comparisons have only 17 paired complete trajectories and do not restore its undefined full schedule.')
        for metric in ['rmse','crps']:
            self.h('Paired '+metric.upper(),2)
            pairs=[p for p in e.pairs if p['metric']==metric and p.get('mean') is not None]
            self.table(['Track','Comparator','Paired n','Mean difference','95% interval'],[[p['track'],LABEL[p['comparator']],p['n'],num(p['mean']), '['+num(p['ci95'][0])+', '+num(p['ci95'][1])+']'] for p in pairs],[40,150,60,115,150])
        self.p('Observed conclusions are configuration- and protocol-specific. In A the Bayesian model has higher observed mean RMSE than each fully reliable comparator, and higher CRPS than both native competitors. Its paired RMSE difference versus ridge ESN is +0.0787 with 95% CI [-0.0237,0.2045], so that paired comparison remains uncertain. In B it loses to CNN, VAR, GRU and GP on paired RMSE, and to GP/GRU on CRPS. Differences against ESN, persistence and mean include zero. The experiment does not establish that Bayesian methods generally fail or that Monte Carlo convergence caused these predictive losses.')
        self.h('9. Error at the six retained forecast leads')
        self.fig('audited_error_leads',465)
        self.p('The full 100-lead files were audited before the outage but are not available in this recovered bundle. These figures use only h=1,5,10,25,50,100, corresponding to physical times 0.25,1.25,2.5,6.25,12.5,25. Joining these points does not reconstruct missing intermediate behavior. Track A/B input access and target sets remain distinct.')
        self.h('10. Probabilistic lead behavior')
        self.fig('audited_probability_leads',565)
        self.p('Bayesian aggregate coverage masks substantial early undercoverage and later overcoverage. This is an observed property of the recorded paths; the plot does not identify a causal mechanism. The GRU is sharper but often undercovers. Point accuracy, calibration, sharpness and proper scores require separate interpretation.',True)
        self.h('11. Aggregate calibration and sharpness')
        self.fig('audited_calibration_width',430)
        self.p('The Bayesian intervals are wider than those of the competing native families. Their coverage advantage in some aggregates does not reverse their worse CRPS. The main probability tables retain confidence intervals and clean-field target definitions, so interval width and coverage can be assessed jointly rather than ranking coverage alone.')
        self.h('12. Predictive performance versus recorded compute')
        self.fig('audited_accuracy_compute',300)
        self.table(['Track / family','Fit seconds','Forecast seconds','Fit-receipt peak MiB'],[[r['track']+' '+SHORT[r['model']],num(r['resource_reconstruction']['fit_seconds'],2),num(r['resource_reconstruction']['forecast_seconds'],2),num(r['resource_reconstruction']['fit_receipt_peak_rss_bytes']/2**20,1)] for r in e.rows],[155,115,115,130])
        self.p('These peak values are process-local getrusage observations from available fit receipts, not complete forecast-worker peaks. Original attempt-level /proc readings were unreliable because of PID namespaces. Total family screening costs and full final external-observer details are not recovered here and are not filled in from estimates.',True)
        self.h('13. Model-seed sensitivity on a matched subset')
        self.p('E2: transcript-recovered means and two retained contrasts. All 56 additional sensitivity fit jobs were recorded complete, with 1,120 finite origins and no observed failure/divergence. Each seed uses the same four trajectories [0,5,10,15]; this includes the main seed, whose four-case mean differs from its twenty-case main result.')
        self.fig('transcript_seed_sensitivity',260)
        sd=e.trans['matched_seed_displayed_results'];mapping={'A_bayes':'A Bayes','A_gru':'A GRU','A_esn':'A ESN','B_bayes':'B Bayes','B_gru':'B GRU','B_esn':'B ESN','B_cnn':'B CNN'}
        self.table(['Matched family','20261105','20261106','20261107'],[[mapping[k]]+[num(v,6) for v in vals] for k,vals in sd['rmse'].items()],[155,120,120,120])
        self.p('CNN alternative-minus-main mean RMSE differences were +0.134034 [−0.077086,0.345154] and +0.172369 [−0.304962,0.649700]. Both intervals include zero; four clusters give limited descriptive evidence. No seed was selected from these outcomes. Seeds are neither posterior draws nor additional trajectories. Full matched per-episode records and the original final matched export are not recovered.')
        self.h('14. Contiguous-site robustness: catastrophic transfer')
        self.p('E2: retained terminal summaries, explicitly lower evidence availability than the hash-verified main audit. Frozen A configurations were transferred without test tuning, with fresh contiguous training transforms and fits. Every method retained a 400-origin denominator.')
        self.table(['Family','Complete n','RMSE [95% CI]','CRPS','90% coverage','90% width'],[[LABEL[r['model']],r['complete_trajectories'],num(r['rmse'])+('\n['+num(r['rmse_ci95'][0],2)+', '+num(r['rmse_ci95'][1],2)+']' if r.get('rmse_ci95') else ''),num(r.get('crps')),num(r.get('coverage90')),num(r.get('width90'))] for r in e.trans['C_displayed_results']],[120,55,100,90,75,95])
        self.fig('transcript_contiguous_outcomes',265)
        self.p('The Bayesian C failure is catastrophic: RMSE about 2106.69, CRPS 905.34 and all 400 origins finite-divergent. Twenty fitted posteriors were recorded qualified. At lead 1, RMSE was already 3607.93 and 90% coverage 13.29%; this is not only late chaotic divergence. Its aggregate 93.99% coverage accompanies width 6877.46 and does not rescue extreme forecast errors. The mean anchor has the lowest defined C RMSE, 1.18437.')
        self.p('GRU has 95/400 finite divergences. NVAR has 20 completed fits but 260/400 nonfinite forecast failures, 140 finite origins (37 divergent), and zero complete trajectories; its aggregate RMSE is unavailable.',True)
        self.h('15. Bounded spatial-representation diagnosis')
        self.p('E2. A post-test read-only diagnosis of the fixed trajectory 0 / first target 8000 compared A and C Bayesian/GRU artifacts. Its historical 48 identity, coordinate, time, transform and input-boundary checks passed. No checked unit/index/partition/seam mismatch was found. No fit, parameter draw, network forward pass or new forecast was performed. The original 39,174-byte diagnostic receipt and complete source manifest are not recovered; the retained numbers below are not a new verification.')
        self.table(['Saved geometry quantity','Interleaved A','Contiguous C'],[['Bayesian training-basis condition number','4.0974e3','5.0771e9'],['Bayesian maximum target basis-row norm','1.43653','74.65577'],['Bayesian retained lambda median','4.43492e-6','2.60232e-7'],['Bayesian weak training singular directions','0','6'],['GRU maximum target basis-row norm','5.20933','1519.97252'],['GRU projection/decoder operator norm','1.16220','529.51647']],[280,118,117])
        self.p('The Bayesian p=32 dictionary uses training-coordinate centering, column standardization, division by sqrt(31), and an intercept. The 1e-6 scale floor was inactive; minimum C raw-column SD was 0.002802607. Weakly identified spatial directions nevertheless have large target loadings. The retained precision summaries describe fitted states, not an altered prior family.')
        self.p('The GRU training basis satisfied Phi_T^T Phi_T/n=I to 2.47e-11 in C. Yet its unseen-gap target basis loadings and geometry-only projection/decoder norm were enormous. This shows why well-conditioned training coordinates alone do not establish stable spatial extrapolation.')
        self.p('The observation supports weak training identification and amplified unseen-gap basis extension as a mechanism consistent with failure. It does not partition error among causes, prove absence of all numerical problems, or establish a universal theorem from one fixed case. It does not justify suppressing the unfavorable C outcomes.')
        self.p('Retained C Bayesian diagnostics: 20/20 qualified (19 initial, one bounded continuation); worst hyperparameter R-hat 1.00870834, minimum bulk ESS 608.05 and tail ESS 744.62. All coefficient/functionals gates were recorded passing. These are transcribed summary extrema, not a recomputation from missing chains. Passing inference diagnostics is compatible with a poorly predictive model.')
        self.h('16. Acceptance, resource accounting and operational limits')
        self.p('E1 preflight established archive/catalog checks, 23 HDF5 external links, chronological masks/transforms, periodic coordinates, hidden/future-label invariance and meaningful score/resume tests. The 72 pilot synchronization cases passed the declared history-128 replay tolerance. Actual core Bayesian forecast invariance covered all eight trajectory-0 candidate/track fits and 16 perturbations; their saved fits remained unchanged. The contiguous history check was post-freeze, training-only, 23/23, not pretest proof of universal synchronization.')
        self.table(['Resource / policy','Recorded evidence'],[['Container vs host','8 GiB cgroup limit; eight CPU quota, nine affinity-visible CPUs; no detected GPU'],['Initial disk / reserve','62,315,782,144 available bytes; 12,463,156,428-byte reserve (20%). A quota command was unavailable'],['Numerical workers','OMP/OpenBLAS/MKL/NumExpr and Torch threads limited to one; expensive pilots sequential; up to two workers only after measured admission'],['Memory admission','Combined configured budget 4,080,218,931 bytes, dynamic 65% headroom gate, 1.5 pilot-RSS multiplier'],['Pilot effort','Short Bayes/GP/GRU/CNN fits about 3.708/1.797/1.537/0.527 seconds; resource profiles, not final scientific results'],['Campaign ceilings','Eight cumulative worker hours; 2.5 for screening. Retained terminal status: 6.033 worker hours [E2]'],['Scientific jobs vs handoff','900 terminal scientific jobs [E2]; complete final artifact package unavailable at recovery. These are separate completion criteria']],[145,370])
        self.p('Three published operational receipts record memory-admission pauses and read-only file-cache advice, with no content changes/deletions. Near the 8 GiB limit, the max counter rose from 5,842 to 5,868 during the second event; it reached 6,450 by the third. OOM/kill counters remained zero. Nonzero max counters document real pressure; zero OOM does not imply no pressure. The first observer aggregate survived, but only six early per-job rows were retained and its full job count was unknown.')
        self.p('E2 final cleanup facts: one guarded pass deleted 21 redundant fit staging files (489.503 MB logical). Canonical fits/receipts were postchecked without changes. Observed available-space gain was 389.915 MB; three reappeared fit staging files and seven archive staging files were retained, totaling 768.801 MB, because repeat deletion was not authorized and descriptor verification remained unresolved. No reappearance mechanism is inferred. These historical quantities do not describe current recovered-workspace usage.')
        self.p('The independent main audit historically reconciled 6,800 origins, 340 episodes, 337 complete horizon records, 75 paired comparisons and 17 examples. Its published receipt records 97,201 numeric and 161,080 metadata checks, zero discrepancies. The missing final extension audit was recorded as passing 65,100 numeric and 321,275 metadata checks across 4,320 origins; only its partial transcript and original hash remain here. No new certification of missing raw inputs is claimed.')
        self.h('17. Provenance, unavailable artifacts and reproduction')
        self.p('Experiment identity: '+f['experiment_identity'],True);self.p('Freeze identity: '+f['freeze_identity'],True)
        self.p('The frozen specification was written at '+f['created_utc']+'; first main-test access was recorded at '+f['first_test_access_utc']+'. Numerical implementation remains the published application branch. Its earlier simulation snapshot and accepted core source are preserved; this recovery renderer is a separate reporting-only program.')
        self.table(['Verified delivered object','Bytes','SHA-256'],[[r['name'],r['bytes'],r['sha256']] for r in e.pre['delivery']['files']],[145,65,305])
        self.p('Only the KS v1 data, source/metadata v2, full catalog and restore helper were authorized for retrieval. Version labels are independent. The published offline verifier proves bytes/hashes, not present account access. Public evidence omits private accounts, collection IDs, authorization receipts and credentials. Original archive hashes above can verify reacquired copies.')
        self.table(['Available','Unavailable or only partially recovered'],[['Published implementation, frozen settings, main audit and six-lead results','Full final result directory and exact extension audit'],['New explicitly labeled C/seed/geometry transcription','Original extension per-episode/raw forecast/checkpoint evidence'],['New report and aggregate figures generated from surviving evidence','A reconstruction of lost 100-lead arrays, space-time fields or path samples'],['Reproducible retrieval/training/report code','A promise that a rerun is free, identical in timing, or already completed']],[255,260])
        self.p('The known lost final extension audit SHA-256 is 77f62520bb50420b3bc24844f3c43b3bfeea42ac47a50571ba03a87ffa8cc98b (665,973 bytes). The recovered report generator is not the lost full report or that audit. Existing full-report identity gates remain intact; no invented summary.json or matched result file is passed to them.')
        if e.custody:
            check=e.custody['main_export_verification']
            self.p('A separate recovery custody check compared the new main extract with the published audit: '+str(check['exact_numeric_leaf_matches'])+' numerical leaves match exactly, with no field or census mismatches. This check verifies faithful extraction and basic arithmetic, not the missing original per-origin records or bootstrap computations. Its input receipt is included in the report manifest.',True)
        self.h('18. Reproducible commands and concrete next actions')
        self.p('The repository documents the original numerical environment in requirements-ks.lock and report environment separately. Large payloads/cache belong outside Git and HOME. Commands below reproduce or resume an actual available run directory; they do not imply that the missing final directory has been reconstructed. Re-running missing fits requires a new explicit resource allocation decision rather than silently exhausting the original eight-hour budget.')
        commands='''export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CACHE_ROOT=/authorized/project/ks-cache
export DOWNLOAD_DIR=/authorized/project/ks-incoming
export RESULTS=/authorized/project/ks-results
export CONFIG=configs/applications/ks_experiment.json
python tools/verify_ks_delivery.py --incoming "$DOWNLOAD_DIR"
python tools/verify_ks_delivery.py --incoming "$DOWNLOAD_DIR" --restore "$CACHE_ROOT"
python -m bayes_stesn.applications.ks_runner screen --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner final --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner sensitivity --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner robustness --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner resume --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner status --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"'''
        for line in commands.splitlines():self.p(line,True)
        self.p('Restore must preserve data/ks/raw and canonical relative paths for HDF5 external links. A missing original checkpoint cannot be resumed merely by copying a hash; recover actual bytes and verify identity first. Completed compatible fits and origin receipts are reused, failures remain explicit, and numerical configuration changes require a new identity.')
        self.h('Render this recovery report only',2)
        self.p('python tools/build_ks_recovery_report.py --repo REPO --recovery TRANSCRIPT_RECOVERY_JSON --output OUTPUT_DIRECTORY --marker-continuity',True)
        self.p('This reporting command opens saved JSON evidence and source references only. It performs no data retrieval, model fitting, forecasting, selection, chain diagnostics, or score recomputation. Recovered fixed-origin example figures, if independently available, must have their own actual-array provenance and be distinguished from the original full run. They cannot be manufactured from aggregate tables.')
        self.p('Deliverables include this PDF, a self-contained HTML version, publication-ready figure PNG/PDF pairs, and a manifest binding inputs, source and generated files. The source of each table/figure is explicitly E1 or E2. Full exact main results remain in the published audit JSON; the HTML appendix includes all 102 retained lead records.')
        if e.example:
            ex=e.example
            self.h('19. One actual fixed-origin illustration: newly recomputed')
            self.p('E3: a new replay from the retained original A Bayesian fit, not recovered original prediction-file bytes. The adapter verifies the same data, frozen configuration, named randomness, history 128, horizon 100, 512 paths and 24 binding checks. All 51 frozen numerical source hashes and the native Bayesian computation/runtime identity match. There is no new fit, inference or selection.')
            self.fig('recomputed_example',355)
            self.p('This case has newly recomputed RMSE 1.3024696676 and CRPS 0.7714993446. Its six retained historical lead RMSE values agree exactly; corresponding interval widths differ by at most 8.88e-16. This is a fixed illustrative case, not a new aggregate replicate or a substitute for the lost full result set.')
            self.p('The recovered environment reproduces the relevant NumPy/SciPy/BLAS Bayesian runtime. It does not reproduce the entire original study runtime: Torch is absent, no neural computation is performed, and the original runner runtime guards remain unchanged. The family-scoped replay is separately identified in reconstruction.json.')
            self.p('A matching 10-second, 100-frame H.264 movie is supplied separately as recovered_example/recomputed_A_bayes_t00_a8000.mp4. The report generator only reads the already-created image and receipts; it performs no forecast. The example remains at the predeclared trajectory/origin, with no case search.',True)
            movie=e.out/'recovered_example/recomputed_A_bayes_t00_a8000.mp4'
            self.blocks.append('<video controls preload="metadata" style="width:100%;max-width:1200px"><source type="video/mp4" src="data:video/mp4;base64,'+base64.b64encode(movie.read_bytes()).decode()+'"></video>')
            self.h('19 continued. Predictive intervals at fixed coordinates')
            self.fig('recomputed_intervals',490)
            self.p('The target positions were fixed before original test access and were retained during recovery. These plots illustrate one origin; they are not evidence of across-trajectory calibration or a replacement for the main score denominators. No additional forecast, scoring, fit or selection was needed to render these saved predictive summaries. The visibly narrow early bands and broad later intervals are consistent with the main audited lead-wise coverage pattern, but one example cannot establish that aggregate result.')
        self.h('20. Primary methodological references')
        for index,(citation,url,code) in enumerate(e.references,1):
            self.p('['+str(index)+'] '+citation,True);self.p(url+(('\nAuthor code / supporting source: '+code) if code else ''),True)
        self.p('These sources motivate model classes and checks. The prepared dataset and frozen protocol govern this experiment. No imported Lyapunov normalization, unmatched leaderboard, solver-generated training input, or claim of state-of-the-art performance is introduced. Technical implementation details are documented in the repository native audit, model registry, protocol and reproduction guide.')
        # HTML appendix only: exact retained values, not fabricated intermediate curves.
        self.blocks.append('<h1>HTML appendix: all 102 retained main lead records (E1)</h1><p>Exactly six selected leads per family/track. CI values are copied from the published audit.</p>')
        heads=['Track','Family','h','Time','n','RMSE / CI','MAE / CI','NRMSE / CI','CRPS / CI','90% coverage / CI','90% width / CI']
        rows=[[r['track'],LABEL[r['model']],r['horizon'],r['elapsed_physical_time'],r['complete_trajectories']]+[est(r,k,5) for k in ['rmse','mae','nrmse','crps','coverage_0.9','width_0.9']] for r in e.leads]
        self.blocks.append('<div class="table-wrap"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in heads)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x)).replace('\n','<br>')+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>')
        self.blocks.append('<h1>Machine-readable input bindings</h1><pre>'+html.escape(json.dumps(e.inputs,indent=2))+'</pre>')
    def render(self):
        out=self.e.out;pdf=out/'ks_audited_recovery_report.pdf';ht=out/'ks_audited_recovery_report.html'
        self.build_content()
        def footer(canvas,doc):
            canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d5dfe3'));canvas.line(40,37,555,37);canvas.setFont(self.font,7);canvas.setFillColor(colors.HexColor('#506471'));canvas.drawString(40,24,'KS audited recovery report | E1 audit / E2 transcript / E3 fixed replay | full archive unavailable');canvas.drawRightString(555,24,str(doc.page));canvas.restoreState()
        pdf_memory=io.BytesIO()
        doc=SimpleDocTemplate(pdf_memory,pagesize=A4,leftMargin=40,rightMargin=40,topMargin=40,bottomMargin=50,title='Kuramoto-Sivashinsky: audited results and recovery report',author='KS research implementation',pageCompression=1)
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        atomic_bytes(pdf,pdf_memory.getvalue())
        css='body{font-family:system-ui,Arial,sans-serif;color:#20313d;background:#edf2f4;margin:0;padding:35px}main{max-width:1200px;margin:auto;padding:40px;background:white;border-radius:6px}h1{color:#17384b;border-top:2px solid #d7e6e9;padding-top:24px;margin-top:40px}h2{color:#207776}p{line-height:1.65}p.small{font-size:.9em;color:#526672}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:.9em}th{background:#214b60;color:white;text-align:left;padding:10px}td{padding:9px;border-bottom:1px solid #dce5e9;vertical-align:top}tr:nth-child(even){background:#f3f7f8}.table-wrap{overflow-x:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:.9em;line-height:1.5;color:#526672}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f5f6;padding:18px}.equation{font-family:monospace}footer{margin-top:40px;font-size:.8em;color:#526672}@media(max-width:750px){body{padding:8px}main{padding:18px}}'
        atomic_bytes(ht,('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KS audited results and recovery report</title><style>'+css+'</style></head><body><main>'+''.join(self.blocks)+'<footer>Report-only reconstruction from explicitly identified surviving evidence. No models executed.</footer></main></body></html>').encode())
        reader=PdfReader(pdf);texts=[p.extract_text() or '' for p in reader.pages]
        required=['artifact handoff incomplete','E1','E2','catastrophic','665,973','512']
        combined='\n'.join(texts)
        for value in required:
            if value not in combined:raise ValueError('Missing narrative requirement: '+value)
        outputs=[]
        for p in [pdf,ht]+[q for a in self.e.figures for q in [a['png'],a['pdf']]]:outputs.append({'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)})
        manifest={'schema':'ks-audited-recovery-report/1','generated_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'freeze_identity':self.e.frozen['freeze_identity'],'inputs':self.e.inputs,'outputs':outputs,'pages':len(reader.pages),'action_counts_scope':'this report renderer; separately identified E3 example was generated by a different verified replay operation','model_fits':0,'forecasts':0,'dataset_opened':False,'score_recomputation':False,'original_full_report_gate_bypassed':False,'original_artifact_handoff_complete':False,'includes_new_single_case_replay':self.e.example is not None,'authoring_marker':'continuity: original report-create marker completed before outage; no duplicate marker','visual_qa_status':'pending_all_page_inspection','limitations':['Main aggregates and selected leads are read from published audit, not reverified from lost raw receipts.','Extension/seed/geometry quantities are explicitly transcript-derived; original full audit not recreated.','No heatmaps, interval traces or raw-path animation fabricated from aggregates.']}
        p=out/'ks_audited_recovery_report_manifest.json';p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'pdf':str(pdf),'html':str(ht),'pages':len(reader.pages),'figures':len(self.e.figures),'manifest':str(p)},indent=2))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--repo',type=Path,required=True);a.add_argument('--recovery',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--marker-continuity',action='store_true');args=a.parse_args()
    if not args.marker_continuity:raise ValueError('This recovery authoring operation requires the already-completed marker continuity acknowledgment')
    e=Evidence(args.repo,args.recovery,args.output);e.figures_build();Report(e).render()
if __name__=='__main__':main()
