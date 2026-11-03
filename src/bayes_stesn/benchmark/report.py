"""Publication-style plots and synchronized animations from saved results only.

No fitting, model selection or random scenario selection occurs while rendering.
Replication zero and the Bayesian ESN are fixed illustration choices, whether
the fit succeeds or fails. Figures distinguish native diagnostic forecasts from
the operational fallback and aggregate independent replications separately.
"""
from __future__ import annotations
from pathlib import Path
import base64
import hashlib
import html
import json
import shutil
import numpy as np

from .contracts import Forecast
from .generators import Simulation
from .runner import write_json
from .. import __version__

COLORS={"persistence":"#94A3B8","seasonal_naive":"#64748B","spatial_persistence":"#94A3B8",
        "basis_lgssm":"#8B5CF6","separable_stgp":"#2563EB","basis_krr":"#D18B22",
        "bayes_esn_ridge":"#008C82","eqesn_rollout":"#E76F51","dcrnn":"#C84B88","basis_gru":"#376B49"}
LABELS={"persistence":"Persistence","seasonal_naive":"Seasonal naive","spatial_persistence":"Spatial persistence",
        "basis_lgssm":"Basis LGSSM","separable_stgp":"Separable GP","basis_krr":"Basis KRR",
        "bayes_esn_ridge":"Bayesian ESN","eqesn_rollout":"Quadratic ESN ensemble","dcrnn":"Diffusion GRU",
        "basis_gru":"Probabilistic basis GRU"}
SCENARIOS={"g1_gaussian_ar":"G1 | Gaussian spatial AR","g2_spectral_transport":"G2 | Spectral transport",
           "g3_excitable_field":"G3 | Nonlinear excitable field","g4_existing_fronts":"G4 | Moving nonlinear fronts",
           "g5_noise_stress":"G5 | Observation/process stress","g6_velocity_shift":"G6 | Hidden velocity shift"}


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":13,
        "axes.labelsize":10,"axes.titleweight":"bold","axes.spines.top":False,
        "axes.spines.right":False,"axes.edgecolor":"#CBD5E1","axes.labelcolor":"#334155",
        "xtick.color":"#64748B","ytick.color":"#475569","text.color":"#10283F",
        "figure.facecolor":"#FFFFFF","axes.facecolor":"#FFFFFF","savefig.facecolor":"#FFFFFF",
        "grid.color":"#E2E8F0","grid.linewidth":.7,"figure.dpi":120,"savefig.dpi":180})
    return plt


def _save(fig,path):
    path=Path(path);fig.savefig(path,bbox_inches="tight",pad_inches=.22)
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight",pad_inches=.22)


def _grid_extent(coordinates):
    """Place image-cell centers at the actual saved grid coordinates."""
    x=np.unique(coordinates[:,0]);y=np.unique(coordinates[:,1])
    dx=x[1]-x[0];dy=y[1]-y[0]
    return (x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2)


def performance_plot(summary,path,*,track="A",metric="rmse",scenario_subset=None):
    plt=_plotting();rows=[r for r in summary["table"] if r["track"]==track]
    if scenario_subset is not None:
        rows=[r for r in rows if r["scenario"] in scenario_subset]
    scenarios=list(dict.fromkeys(r["scenario"] for r in rows))
    if not scenarios:
        return None
    if not any(row.get(metric) is not None for row in rows):
        return None
    fig,axes=plt.subplots(1,len(scenarios),figsize=(5.2*len(scenarios)+1,6),squeeze=False)
    models=list(dict.fromkeys(r["model_id"] for r in rows))
    for ax,scenario in zip(axes[0],scenarios):
        by_model={r["model_id"]:r for r in rows if r["scenario"]==scenario}
        for j,model in enumerate(models):
            row=by_model.get(model,{})
            value=row.get(metric);se=row.get(metric+"_mcse")
            if value is not None:
                ax.errorbar(value,j,xerr=None if se is None else 2*se,fmt="o",markersize=7,
                            capsize=3,color=COLORS.get(model,"#334155"),lw=1.8)
                tag=f"{row['qualified']}/{row['attempted']}"
                ax.annotate(tag,(value,j),xytext=(8,6),textcoords="offset points",fontsize=7,color="#64748B")
            else:
                ax.text(.03,j,"unqualified / no native law",transform=ax.get_yaxis_transform(),fontsize=8,color="#94A3B8")
        ax.set_yticks(range(len(models)),[LABELS.get(m,m) for m in models]);ax.invert_yaxis()
        ax.grid(axis="x");ax.set_axisbelow(True);ax.set_xlabel(f"{metric.upper()}  /  smaller is better")
        ax.set_title(SCENARIOS.get(scenario,scenario),loc="left",fontsize=11,pad=18)
        ax.margins(x=.2)
    if len(scenarios)>1:
        for ax in axes[0][1:]:
            ax.set_yticklabels([])
    domain="observed sites" if track=="A" else "never-observed sites"
    title=f"{metric.upper()} at {domain}"
    fig.suptitle(title,x=.055,y=.995,ha="left",fontsize=21,fontweight="bold")
    detail="All attempts, including declared point fallback" if metric=="rmse" else "Qualified native probabilistic fits only; populations can differ"
    fig.text(.055,.015,f"{detail}. Whiskers: +/-2 Monte Carlo SE across independent datasets. Labels: qualified / attempted.",fontsize=9,color="#64748B")
    fig.tight_layout(rect=(0,.045,1,.94));_save(fig,path);plt.close(fig)
    return str(path)


def lead_plot(run,summary,path,*,scenario_subset=None):
    plt=_plotting();scenarios=list(dict.fromkeys(r["scenario"] for r in summary["table"]))
    if scenario_subset is not None:
        scenarios=[s for s in scenarios if s in scenario_subset]
    fig,axes=plt.subplots(len(scenarios),2,figsize=(13,3.4*len(scenarios)+1.3),squeeze=False)
    handles={}
    for i,scenario in enumerate(scenarios):
        for j,track in enumerate(("A","B")):
            ax=axes[i,j]
            for folder in sorted((Path(run)/scenario/"rep_000"/track).glob("*")):
                receipt=folder/"receipt.json"
                if not receipt.exists():
                    continue
                row=json.loads(receipt.read_text());values=row["operational_scores"]["rmse_by_lead"]
                model=row["model_id"]
                line,=ax.plot(np.arange(1,len(values)+1),values,color=COLORS.get(model),
                             lw=2 if model=="bayes_esn_ridge" else 1.25,alpha=.9,
                             ls="--" if row["fallback"] else "-",label=LABELS.get(model,model))
                handles[model]=line
            ax.set_title(f"{SCENARIOS.get(scenario,scenario)} | {'observed' if track=='A' else 'new sites'}",loc="left",fontsize=10)
            ax.set_xlabel("Forecast lead");ax.set_ylabel("RMSE");ax.grid(alpha=.8)
    fig.suptitle("How forecast error grows with lead",x=.065,y=.995,ha="left",fontsize=21,fontweight="bold")
    fig.legend(handles.values(),[LABELS.get(m,m) for m in handles],loc="lower center",bbox_to_anchor=(.5,.005),ncol=4,frameon=False,fontsize=8)
    caption_y=.14 if len(scenarios)==1 else .095
    fig.text(.065,caption_y,"Preselected replication 0. Dashed lines indicate operational fallback. These curves are illustrations, not aggregate rankings.",fontsize=9,color="#64748B")
    fig.tight_layout(rect=(0,.24 if len(scenarios)==1 else .16,1,.93));_save(fig,path);plt.close(fig)


def data_montage(sim,path):
    plt=_plotting();n=int(round(np.sqrt(len(sim.coordinates))))
    if n*n!=len(sim.coordinates):
        raise ValueError("Data montage currently expects the saved rectangular evaluation grid")
    indices=np.linspace(0,len(sim.observation)-1,5,dtype=int)
    lo,hi=np.quantile(sim.observation,[.005,.995])
    fig,axes=plt.subplots(1,5,figsize=(14,3.5))
    for ax,t in zip(axes,indices):
        im=ax.imshow(sim.observation[t].reshape(n,n),origin="lower",extent=_grid_extent(sim.coordinates),
                     vmin=lo,vmax=hi,cmap="viridis",interpolation="nearest")
        ax.set_title(f"time {sim.times[t]:g}",loc="left",fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,shrink=.67,pad=.015,label="Noisy observation")
    fig.suptitle(SCENARIOS.get(sim.metadata["scenario"],sim.metadata["scenario"]),x=.125,y=1.02,ha="left",fontsize=18,fontweight="bold")
    fig.text(.125,.035,"Shared display scale across time; 0.5-99.5% limits. Grid cells shown directly, without spatial smoothing.",fontsize=9,color="#64748B")
    _save(fig,path);plt.close(fig)


def data_animation(sim,path,*,maximum_frames=100):
    """Animate a fixed uniform temporal subsample; all full-resolution data remain saved."""
    plt=_plotting();from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
    n=int(round(np.sqrt(len(sim.coordinates))))
    indices=np.unique(np.linspace(0,len(sim.observation)-1,min(maximum_frames,len(sim.observation)),dtype=int))
    lo,hi=np.quantile(sim.observation,[.005,.995])
    fig,ax=plt.subplots(figsize=(6.8,6.3));fig.subplots_adjust(left=.12,right=.82,bottom=.16,top=.8)
    im=ax.imshow(sim.observation[indices[0]].reshape(n,n),origin="lower",extent=_grid_extent(sim.coordinates),
                 cmap="viridis",vmin=lo,vmax=hi,interpolation="nearest")
    o=sim.coordinates[sim.observed_indices]
    ax.scatter(o[:,0],o[:,1],s=14,facecolors="none",edgecolors="white",linewidths=.7,label="Observed inputs O")
    ax.set(xlabel="Spatial coordinate x",ylabel="Spatial coordinate y")
    fig.colorbar(im,ax=ax,fraction=.045,pad=.04,label="Noisy observation")
    fig.suptitle(SCENARIOS[sim.metadata["scenario"]],x=.08,y=.97,ha="left",fontsize=16,fontweight="bold")
    title=fig.text(.08,.88,f"time {sim.times[indices[0]]:g}",fontsize=11,color="#64748B")
    ax.legend(loc="upper right",fontsize=8,framealpha=.8)
    fig.text(.08,.035,f"Preselected replication 0. {len(indices)} uniformly selected frames from {len(sim.observation)} saved times.\nFixed 0.5-99.5% color limits. Circle outlines mark observed inputs; other sites are not model histories.",fontsize=8,color="#64748B")
    def update(frame):
        t=indices[frame];im.set_data(sim.observation[t].reshape(n,n))
        title.set_text(f"time {sim.times[t]:g}  |  row {t}  |  frame {frame+1}/{len(indices)}")
        return im,title
    movie=FuncAnimation(fig,update,frames=len(indices),interval=100,blit=False)
    path=Path(path);outputs={"frame_indices":indices.tolist()}
    if shutil.which("ffmpeg"):
        movie.save(path.with_suffix(".mp4"),writer=FFMpegWriter(fps=10,bitrate=1200,extra_args=["-pix_fmt","yuv420p"]),dpi=110)
        outputs["mp4"]=str(path.with_suffix(".mp4"))
    movie.save(path.with_suffix(".gif"),writer=PillowWriter(fps=10),dpi=85)
    outputs["gif"]=str(path.with_suffix(".gif"));plt.close(fig)
    return outputs


def forecast_media(sim,forecast,origin,path,*,qualified=True,model_id="bayes_esn_ridge",track="B",animate=True):
    """Truth / mean / signed error share target sites and synchronized lead."""
    plt=_plotting();from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
    mapping={identity:i for i,identity in enumerate(sim.site_ids)}
    targets=np.array([mapping[x] for x in forecast.site_ids])
    truth=sim.observation[origin:origin+len(forecast.mean),targets]
    error=forecast.mean-truth
    lo,hi=np.quantile(np.r_[truth.ravel(),forecast.mean.ravel()],[.005,.995])
    limit=max(float(np.quantile(np.abs(error),.995)),1e-8)
    n=int(round(np.sqrt(len(sim.coordinates))))
    full_grid=n*n==len(sim.coordinates)
    values=(truth,forecast.mean,error)
    fig=plt.figure(figsize=(12,6.8));grid=fig.add_gridspec(2,3,height_ratios=(3.6,1.35),hspace=.43,wspace=.16)
    axes=[fig.add_subplot(grid[0,j]) for j in range(3)];trace=fig.add_subplot(grid[1,:])
    artists=[]
    def field(v):
        out=np.full(n*n,np.nan);out[targets]=v;return out.reshape(n,n)
    for j,(ax,title) in enumerate(zip(axes,("Actual noisy observation","Forecast mean","Signed error: forecast - actual"))):
        cmap="RdBu_r" if j==2 else "viridis";vmin,vmax=(-limit,limit) if j==2 else (lo,hi)
        if full_grid:
            artist=ax.imshow(field(values[j][0]),origin="lower",extent=_grid_extent(sim.coordinates),interpolation="nearest",cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            artist=ax.scatter(forecast.coordinates[:,0],forecast.coordinates[:,1],c=values[j][0],cmap=cmap,vmin=vmin,vmax=vmax,s=36)
        artists.append(artist);ax.set_title(title,fontsize=10,loc="left",pad=10)
        ax.set_xticks(np.linspace(sim.coordinates[:,0].min(),sim.coordinates[:,0].max(),3))
        ax.set_yticks(np.linspace(sim.coordinates[:,1].min(),sim.coordinates[:,1].max(),3));ax.set_facecolor("#EEF2F6")
        fig.colorbar(artist,ax=ax,shrink=.78,pad=.025,fraction=.045)
    lead=np.arange(1,len(truth)+1);rmse=np.sqrt(np.mean(error**2,axis=1))
    trace.plot(lead,rmse,color=COLORS[model_id],lw=2.4,label="RMSE at each lead")
    running=np.sqrt(np.cumsum(np.mean(error**2,axis=1))/lead)
    trace.plot(lead,running,color="#10283F",lw=1.3,ls="--",label="RMSE through current lead")
    cursor=trace.axvline(1,color="#C84B88",lw=1.2);dot,=trace.plot([1],[rmse[0]],"o",color="#C84B88",ms=5)
    trace.set(xlabel="Forecast lead",ylabel="Prediction error",xlim=(1,max(2,len(truth))));trace.grid()
    trace.legend(loc="upper left",frameon=False,ncol=2,fontsize=8)
    status="native qualified fit" if qualified else "UNQUALIFIED FIT: diagnostic illustration only"
    heading=fig.suptitle(f"{SCENARIOS[sim.metadata['scenario']]}  |  {LABELS[model_id]}",x=.07,y=.995,ha="left",fontsize=17,fontweight="bold")
    subtitle=fig.text(.07,.94,f"lead 1 / {len(truth)}  |  {status}",fontsize=10,color="#64748B")
    fig.text(.07,.012,f"Replication 0, track {track}. Fixed forecast origin {origin}; no future assimilation. Gray cells are outside the scored target domain.\nDisplay limits fixed across all frames (0.5-99.5%); error colors centered at zero.",fontsize=8,color="#64748B")
    fig.subplots_adjust(top=.87,bottom=.12,left=.07,right=.96)
    def update(index):
        for j,artist in enumerate(artists):
            if full_grid:
                artist.set_data(field(values[j][index]))
            else:
                artist.set_array(values[j][index])
        cursor.set_xdata([index+1,index+1]);dot.set_data([index+1],[rmse[index]])
        subtitle.set_text(f"lead {index+1} / {len(truth)}  |  time {sim.times[origin+index]:g}  |  {status}")
        return artists+[cursor,dot,subtitle]
    middle=min(len(truth)-1,max(0,len(truth)//2));update(middle)
    path=Path(path);_save(fig,path.with_suffix(".png"))
    outputs={"still":str(path.with_suffix(".png"))}
    if animate:
        frames=FuncAnimation(fig,update,frames=len(truth),interval=100,blit=False)
        if shutil.which("ffmpeg"):
            movie=path.with_suffix(".mp4")
            frames.save(movie,writer=FFMpegWriter(fps=10,bitrate=1800,extra_args=["-pix_fmt","yuv420p"]),dpi=110)
            outputs["mp4"]=str(movie)
        gif=path.with_suffix(".gif")
        frames.save(gif,writer=PillowWriter(fps=10),dpi=80);outputs["gif"]=str(gif)
    plt.close(fig)
    return outputs


def _inline_png(path):
    return "data:image/png;base64,"+base64.b64encode(Path(path).read_bytes()).decode()


def _pdf_report(output,summary,manifest,figures):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak,Table,TableStyle
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.utils import ImageReader
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name="Hero",fontName="Helvetica-Bold",fontSize=29,leading=33,textColor=colors.HexColor("#10283F"),spaceAfter=20))
    styles.add(ParagraphStyle(name="Deck",fontName="Helvetica",fontSize=13,leading=19,textColor=colors.HexColor("#475569"),spaceAfter=14))
    styles["BodyText"].fontSize=10;styles["BodyText"].leading=15
    width,height=landscape(A4);margin=42;usable=width-2*margin
    doc=SimpleDocTemplate(str(output),pagesize=(width,height),rightMargin=margin,leftMargin=margin,topMargin=margin,bottomMargin=40,
        title="Forecasting spatial dynamics: model comparison evidence",author="bayes-stesn contributors")
    attempted=sum(r["attempted"] for r in summary["table"]);qualified=sum(r["qualified"] for r in summary["table"])
    story=[Paragraph("Forecasting spatial dynamics",styles["Hero"]),
           Paragraph("Bayesian ESNs and comparable models<br/>Implementation and reproducible benchmark evidence",styles["Deck"]),
           Paragraph(f"Stage: <b>{html.escape(summary['stage'])}</b>. {html.escape(summary['comparison_status'])}.",styles["BodyText"]),Spacer(1,18),
           Paragraph(f"<b>{attempted}</b> final model/task attempts &nbsp;&nbsp; <b>{qualified}</b> qualified fits &nbsp;&nbsp; <b>{attempted-qualified}</b> explicit failures or unqualified fits",styles["Deck"]),
           Paragraph("Point comparisons include every attempted fit through a declared persistence fallback. Probability scores use qualified native laws, with their populations reported. Intervals summarize variability across independent datasets; sites and time steps do not count as extra replications.",styles["BodyText"]),Spacer(1,15),
           Paragraph(f"Protocol: {html.escape(manifest['protocol']['name'])}; forecast horizon {manifest['protocol']['horizon']}; observed sites {manifest['protocol']['n_observed']}; independent datasets per scenario {manifest['protocol']['replications']}.",styles["BodyText"]),
           Paragraph(f"Run identity: {manifest['identity'][:24]}<br/>Source fingerprint: {manifest['source_fingerprint'][:24]}",styles["Code"]),
           Paragraph("The animations show a fixed, preselected replication. Unqualified native forecasts are prominently labeled diagnostic; they are never silently counted as qualified probabilistic results.",styles["BodyText"])]
    for title,path in figures:
        story.extend([PageBreak(),Paragraph(html.escape(title),styles["Heading1"]),Spacer(1,8)])
        iw,ih=ImageReader(path).getSize();scale=min(usable/iw,(height-150)/ih)
        story.append(Image(str(path),width=iw*scale,height=ih*scale))
    story.extend([PageBreak(),Paragraph("Numerical values and interpretation",styles["Heading1"])])
    table=[["Scenario / task","Model","Qualified","RMSE","CRPS","90% coverage"]]
    for row in summary["table"]:
        def value(key):
            return "--" if row.get(key) is None else f"{row[key]:.3f}"
        table.append([row["scenario"].split("_")[0].upper()+" / "+row["track"],LABELS[row["model_id"]],
                      f"{row['qualified']}/{row['attempted']}",value("rmse"),value("crps"),value("coverage_90")])
    tab=Table(table,colWidths=[91,195,75,80,80,90],repeatRows=1)
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#10283F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F5F9")]),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ALIGN",(2,0),(-1,-1),"RIGHT")]))
    story.append(tab)
    def footer(canvas,document):
        canvas.setStrokeColor(colors.HexColor("#E2E8F0"));canvas.line(margin,30,width-margin,30)
        canvas.setFillColor(colors.HexColor("#64748B"));canvas.setFont("Helvetica",8)
        canvas.drawString(margin,18,"bayes-stesn "+__version__+" | saved-artifact report | "+summary["stage"])
        canvas.drawRightString(width-margin,18,str(document.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def create_report(run,output,*,animations=True):
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    figures_dir=output/"figures";media_dir=output/"media";figures_dir.mkdir(exist_ok=True);media_dir.mkdir(exist_ok=True)
    summary=json.loads((run/"summary.json").read_text());manifest=json.loads((run/"manifest.json").read_text())
    if manifest.get("schema")=="benchmark-run-4.0" and (
            not summary.get("complete") or (run/"invalidation.json").exists()):
        raise ValueError("Complete, provenance-verified scientific outcomes are required for this benchmark report")
    figures=[];media=[];data_media=[]
    scenarios=list(dict.fromkeys(r["scenario"] for r in summary["table"]))
    chunks=[scenarios[i:i+2] for i in range(0,len(scenarios),2)]
    for track in ("A","B"):
        for metric in ("rmse","crps"):
            for j,chunk in enumerate(chunks):
                path=figures_dir/f"{track}_{metric}_{j+1}.png"
                if performance_plot(summary,path,track=track,metric=metric,scenario_subset=chunk):
                    figures.append((f"Track {track}: {metric.upper()} - "+", ".join(s.split("_")[0].upper() for s in chunk),path))
    for j,chunk in enumerate(chunks):
        path=figures_dir/f"error_by_lead_{j+1}.png"
        lead_plot(run,summary,path,scenario_subset=chunk);figures.append(("Error evolution: preselected replication",path))
    for scenario in scenarios:
        base=run/scenario/"rep_000";sim=Simulation.load(base/"simulation.npz")
        path=figures_dir/f"{scenario}_data.png";data_montage(sim,path);figures.append((SCENARIOS[scenario],path))
        if animations:
            data_media.append({"scenario":scenario,**data_animation(sim,media_dir/f"{scenario}_data")})
        for track in ("A","B"):
            model_id="bayes_esn_ridge";folder=base/track/model_id
            if not (folder/"forecast.npz").exists():
                continue
            forecast=Forecast.load(folder/"forecast.npz");receipt=json.loads((folder/"receipt.json").read_text())
            paths=forecast_media(sim,forecast,receipt["origin_index"],media_dir/f"{scenario}_{track}_forecast",
                                qualified=receipt["qualified"],track=track,model_id=model_id,animate=animations)
            media.append({"scenario":scenario,"track":track,"qualified":receipt["qualified"],**paths})
            figures.append((f"{SCENARIOS[scenario]} - track {track} forecast",Path(paths["still"])))
    attempted=sum(r["attempted"] for r in summary["table"]);qualified=sum(r["qualified"] for r in summary["table"])
    sections="".join(f'<section><h2>{html.escape(title)}</h2><img src="{_inline_png(path)}" alt="{html.escape(title)}"></section>' for title,path in figures)
    videos="".join(f'<section><h2>{html.escape(SCENARIOS[m["scenario"]])} / track {m["track"]}</h2><p>{"Qualified native fit" if m["qualified"] else "Unqualified native fit: diagnostic illustration only"}</p><video controls loop muted playsinline src="media/{Path(m["mp4"]).name}"></video></section>' for m in media if "mp4" in m)
    videos="".join(f'<section><h2>{html.escape(SCENARIOS[m["scenario"]])} / data evolution</h2><video controls loop muted playsinline src="media/{Path(m["mp4"]).name}"></video></section>' for m in data_media if "mp4" in m)+videos
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Forecasting spatial dynamics</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f1f5f9;color:#10283f}}main{{max-width:1180px;margin:auto;padding:48px 24px}}header{{padding:40px;border-radius:20px;background:#10283f;color:white}}h1{{font-size:48px;line-height:1.05;max-width:780px;margin:12px 0 24px}}.eyebrow{{color:#72d8cd;letter-spacing:.18em;text-transform:uppercase;font-size:12px}}.cards{{display:flex;gap:18px;margin:24px 0;flex-wrap:wrap}}.card{{padding:20px 28px;background:white;border-radius:14px;flex:1}}.card b{{font-size:32px;display:block}}section{{margin:28px 0;padding:24px;background:white;border-radius:16px}}h2{{font-size:23px}}img,video{{max-width:100%;height:auto}}.note{{color:#64748b}}code{{overflow-wrap:anywhere;font-size:12px}}a{{color:#008c82}}</style>
<main><header><div class="eyebrow">Reproducible spatiotemporal forecasting / version {__version__}</div><h1>Forecasting spatial dynamics</h1><p>Bayesian ESNs, classical state-space and kernel models, reservoir ensembles, and neural comparators.</p><p>{html.escape(summary['comparison_status'])}</p></header>
<div class="cards"><div class="card"><b>{attempted}</b>final model/task attempts</div><div class="card"><b>{qualified}</b>qualified fits</div><div class="card"><b>{manifest['protocol']['replications']}</b>independent datasets per scenario</div></div>
<section><h2>How to read this evidence</h2><p>Point scores include every attempt with an explicit persistence fallback for failed fits. Probability scores use qualified native distributions and report their populations. Whiskers show twice the Monte Carlo standard error across independent datasets. Fit seeds, sites, and forecast leads are not extra dataset replications.</p><p class="note">Animations use replication 0, selected before evaluation. Gray cells lie outside the target domain. Forecasts run freely from one fixed origin; no future observations are assimilated. Unqualified forecasts are labeled diagnostic. The finite-grid excitable-field DGP is not evidence of continuum mesh convergence.</p></section>
{sections}{videos}<section><h2>Reproduction</h2><p>Run identity <code>{manifest['identity']}</code><br>Source fingerprint <code>{manifest['source_fingerprint']}</code></p><p>Numerical records: <a href="summary.json">summary.json</a>. Full model checkpoints, simulation arrays, per-candidate costs and failure traces are in the run directory.</p></section></main></html>'''
    (output/"report.html").write_text(text,encoding="utf-8")
    write_json(output/"summary.json",summary);write_json(output/"run_manifest.json",manifest)
    _pdf_report(output/"benchmark_report.pdf",summary,manifest,figures)
    inventory=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name!="report_manifest.json":
            inventory.append({"path":str(path.relative_to(output)),"bytes":path.stat().st_size,
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt={"report":str(output/"report.html"),"pdf":str(output/"benchmark_report.pdf"),
             "run_identity":manifest["identity"],"render_only":True,"media":media,"data_media":data_media,"files":inventory}
    write_json(output/"report_manifest.json",receipt)
    return receipt
