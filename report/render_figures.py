#!/usr/bin/env python3
"""Render verified scientific figures/media from saved, protocol-locked results.

This renderer never fits a model, changes selection or creates evaluation data.
The illustration is the prespecified matched-scenario replication, using its
already locked validation selection. Every output records source/artifact hashes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayes_stesn_matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors, ticker
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, CORAL, GOLD = "#14283b", "#007f88", "#e76750", "#c5a056"
GRAY, LIGHT, GRID = "#647588", "#f3f6f7", "#dbe3e8"
MODEL_NAMES = {"ridge_short": "Ridge / shorter history", "ridge_long": "Ridge / longer history",
               "ridge_expressive_long": "Ridge / larger reservoir", "group_rhs_expressive_long": "Grouped RHS / larger reservoir",
               "group_gaussian_expressive_long": "Grouped Gaussian / larger reservoir"}
MODEL_SHORT_NAMES = {"ridge_short": "Ridge\nshorter", "ridge_long": "Ridge\nlonger",
                     "ridge_expressive_long": "Ridge\nlarger", "group_rhs_expressive_long": "Grouped RHS\nlarger",
                     "group_gaussian_expressive_long": "Grouped Gaussian\nlarger"}
SCENARIO_NAMES = {"matched_gaussian": "Gaussian innovations", "anisotropic_gaussian": "Anisotropic innovations"}
FIELD_CMAP = colors.LinearSegmentedColormap.from_list("stesn_field", [NAVY, "#488c9d", "#f7f5ef", "#ddba88", CORAL])
ERROR_CMAP = colors.LinearSegmentedColormap.from_list("stesn_error", [TEAL, "#a5cfce", "#fffdfa", "#f3b1a1", CORAL])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": NAVY,
    "axes.labelcolor": NAVY, "axes.edgecolor": GRID, "axes.titleweight": "semibold",
    "axes.titlesize": 12, "axes.labelsize": 10.5, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white", "grid.color": GRID,
    "grid.linewidth": .6, "lines.linewidth": 1.8, "legend.frameon": False,
    "legend.fontsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def heading(fig, title, subtitle, *, top=.975):
    fig.text(.055, top, title, ha="left", va="top", fontsize=20, weight="bold", color=NAVY)
    fig.text(.055, top-.045, subtitle, ha="left", va="top", fontsize=10.5, color=GRAY)


def footer(fig, text, y=.025):
    fig.text(.055, y, text, ha="left", va="bottom", fontsize=8.5, color=GRAY)


def style_axis(ax, *, grid=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", alpha=.85)
    ax.tick_params(length=0, pad=6)


class Renderer:
    def __init__(self, study, output, fps=10):
        self.study, self.output, self.fps = Path(study), Path(output), int(fps)
        self.figures, self.media = self.output/"figures", self.output/"media"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.media.mkdir(parents=True, exist_ok=True)
        self.inputs, self.artifacts, self.warnings = {}, {}, []
        self.protocol = self.summary = None
        self.cases = []
        self.illustration = None

    def source(self, path):
        path = Path(path)
        resolved = path.resolve()
        try:
            identity = str(resolved.relative_to(ROOT))
        except ValueError:
            identity = str(resolved)
        self.inputs[identity] = sha256(path)
        return path

    def save_figure(self, fig, name, meaning):
        path = self.figures/name
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)
        with Image.open(path) as im:
            size = list(im.size)
        self.artifacts[name] = {"file": str(path.relative_to(self.output)), "sha256": sha256(path),
                                 "pixel_dimensions": size, "dpi": 220, "meaning": meaning}
        print(f"figure {name}", flush=True)

    def load_study(self, *, illustration_only=False):
        self.protocol = read_json(self.source(self.study/"protocol.json"))
        self.model_order = tuple(self.protocol["config"]["candidate_names"])
        if illustration_only:
            replication = int(self.protocol["illustration_replication"])
            items = [{"path":f"final/matched_gaussian/replication_{replication:03d}/result.json"}]
        else:
            self.summary = read_json(self.source(self.study/"summary.json"))
            if self.summary["protocol_sha256"] != self.protocol["protocol_sha256"]:
                raise ValueError("summary/protocol identity mismatch")
            items = self.summary["case_results"]
        for item in items:
            case = read_json(self.source(self.study/item["path"]))
            if case["protocol_sha256"] != self.protocol["protocol_sha256"]:
                raise ValueError("case/protocol identity mismatch")
            self.cases.append(case)
        if not illustration_only:
            expected = {(scenario, replication)
                        for scenario in self.protocol["scenarios"]
                        for replication in range(int(self.protocol["replications"]))}
            observed = [(case["scenario"], int(case["replication"])) for case in self.cases]
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError("final summary must contain every prespecified case exactly once")
        replication = int(self.protocol["illustration_replication"])
        directory = self.study/"final"/"matched_gaussian"/f"replication_{replication:03d}"
        match = [c for c in self.cases if c["scenario"] == "matched_gaussian" and c["replication"] == replication]
        if len(match) != 1:
            self.warnings.append("Prespecified matched-scenario illustration case is absent; no replacement was selected.")
            return
        if match[0]["selection"]["selected"] is None or not (directory/"illustration.npz").exists():
            self.warnings.append("Prespecified illustration has no validated selected model; no forecast animation fabricated.")
            return
        self.source(directory/"selection.json")
        with np.load(self.source(directory/"illustration.npz"), allow_pickle=False) as data:
            self.illustration = {key: np.array(data[key]) for key in data.files}
        with np.load(self.source(directory/"forecast_summaries.npz"), allow_pickle=False) as data:
            if not np.array_equal(data["observation_truth"], self.illustration["observation_truth"]):
                raise ValueError("illustration and final summaries have different observations")
            if not np.array_equal(data["coordinates"], self.illustration["coordinates"]):
                raise ValueError("illustration and final summary domains differ")
        d = self.illustration
        if str(d["selected_model"]) != match[0]["selection"]["selected"]:
            raise ValueError("illustration did not use its locked validation selection")
        if str(d["protocol_sha256"]) != self.protocol["protocol_sha256"]:
            raise ValueError("illustration protocol mismatch")
        if d["mean"].shape != d["observation_truth"].shape or not np.isfinite(d["mean"]).all():
            raise ValueError("invalid saved forecast")
        if not np.allclose(d["mean"], d["observation_paths"].mean(axis=0), rtol=1e-12, atol=1e-12):
            raise ValueError("saved point forecast is not the predictive observation mean")
        q = np.quantile(d["observation_paths"], [.05, .95], axis=0)
        if not np.allclose(q, np.stack((d["lower90"], d["upper90"])), rtol=1e-12, atol=1e-12):
            raise ValueError("saved 90% intervals disagree with the posterior paths")
        d["error"] = d["mean"]-d["observation_truth"]
        d["rmse_by_lead"] = np.sqrt(np.mean(d["error"]**2, axis=1))
        d["covered90"] = (d["observation_truth"] >= d["lower90"]) & (d["observation_truth"] <= d["upper90"])
        d["coverage_by_lead"] = d["covered90"].mean(axis=1)
        d["running_rmse"] = np.sqrt(np.cumsum(np.mean(d["error"]**2, axis=1))/np.arange(1, len(d["mean"])+1))
        d["running_coverage"] = np.cumsum(d["coverage_by_lead"])/np.arange(1, len(d["mean"])+1)
        d["field_limit"] = float(np.max(np.abs(np.r_[d["observation_truth"].ravel(), d["mean"].ravel()])))
        d["error_limit"] = max(float(np.abs(d["error"]).max()), 1e-12)
        x, y = np.unique(d["coordinates"][:, 0]), np.unique(d["coordinates"][:, 1])
        if len(x)*len(y) != len(d["coordinates"]):
            raise ValueError("this grid renderer requires the study's complete Cartesian target grid")
        lookup = {tuple(point): i for i, point in enumerate(d["coordinates"])}
        d["grid_indices"] = np.array([[lookup[(xx, yy)] for xx in x] for yy in y])
        d["extent"] = (x[0]-(x[1]-x[0])/2, x[-1]+(x[-1]-x[-2])/2,
                       y[0]-(y[1]-y[0])/2, y[-1]+(y[-1]-y[-2])/2)
        self.illustration_identity = {"scenario": "matched_gaussian", "replication": replication,
            "selected_model": str(d["selected_model"]), "forecast_origin": int(d["origin"]),
            "horizon": len(d["mean"]), "posterior_paths": len(d["observation_paths"]),
            "posterior_identity": match[0]["models"][str(d["selected_model"])]["posterior_identity"],
            "selection_rule": "saved validation-only choice; no renderer model selection",
            "error_definition": "forecast observation mean minus actual observation",
            "field_color_limits": [-d["field_limit"], d["field_limit"]],
            "error_color_limits": [-d["error_limit"], d["error_limit"]],
            "forecast_mean_check_max_abs": float(np.abs(d["mean"]-d["observation_paths"].mean(axis=0)).max())}

    def grid_image(self, ax, value, *, error=False, absolute=False, limit=None):
        d = self.illustration
        lim = (d["error_limit"] if error or absolute else d["field_limit"]) if limit is None else limit
        cmap = "magma" if absolute else ERROR_CMAP if error else FIELD_CMAP
        image = ax.imshow(value[d["grid_indices"]], origin="lower", extent=d["extent"],
                          cmap=cmap, vmin=0 if absolute else -lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.tick_params(length=0, labelsize=9)
        ax.set_xlabel("Spatial x", labelpad=3)
        ax.set_ylabel("Spatial y", labelpad=3)
        return image

    def montage(self):
        d = self.illustration
        H = len(d["mean"])
        leads = [0, max(0, H//4-1), max(0, H//2-1), H-1]
        fig, axes = plt.subplots(3, 4, figsize=(13.2, 10.0))
        fig.subplots_adjust(left=.065, right=.89, bottom=.11, top=.86, hspace=.34, wspace=.26)
        heading(fig, "Forecasting an evolving spatial field", f"{MODEL_NAMES[str(d['selected_model'])]} · Gaussian scenario · replication 0 · one uninterrupted {H}-step forecast")
        images = []
        for row, (label, key) in enumerate((("Observed", "observation_truth"), ("Forecast mean", "mean"), ("Signed error", "error"))):
            for col, t in enumerate(leads):
                im = self.grid_image(axes[row, col], d[key][t], error=row == 2)
                if row == 0:
                    axes[row, col].set_title(f"Lead {t+1}  ·  t={d['times'][t]:g}", fontsize=11, pad=10)
                if col == 0:
                    axes[row, col].text(-.37, .5, label, transform=axes[row, col].transAxes,
                                      rotation=90, va="center", ha="center", fontsize=12, weight="bold", color=NAVY)
                if row < 2:
                    axes[row, col].set_xlabel("")
                if col > 0:
                    axes[row, col].set_ylabel("")
                images.append(im)
        cax = fig.add_axes([.92, .405, .016, .445])
        fig.colorbar(images[0], cax=cax, label="Observation / forecast · response units")
        cax2 = fig.add_axes([.92, .12, .016, .20])
        fig.colorbar(images[-1], cax=cax2, label="Forecast − observation")
        footer(fig, "Identical fixed scales for observations and forecasts at every lead. Error is signed; teal = underprediction, coral = overprediction.")
        self.save_figure(fig, "forecast_montage.png", "Observed field, predictive observation mean and signed forecast-minus-observation error at four prespecified horizon fractions.")

    def scores(self):
        scenarios = self.protocol["scenarios"]
        fig, axes = plt.subplots(len(scenarios), 2, figsize=(14.4, 8.7), squeeze=False)
        fig.subplots_adjust(left=.23, right=.97, top=.83, bottom=.11, hspace=.43, wspace=.25)
        heading(fig, "Forecast accuracy across independent replications", "Dots = replication means; large markers = mean of eligible replications; intervals = 95% replication bootstrap")
        legend = [Line2D([], [], color=TEAL, marker="o", linestyle="", label="Input sites"),
                  Line2D([], [], color=CORAL, marker="D", linestyle="", label="Held-out sites")]
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.97, .935), ncol=2)
        for row, scenario in enumerate(scenarios):
            cases = sorted([c for c in self.cases if c["scenario"] == scenario], key=lambda c: c["replication"])
            for col, metric in enumerate(("rmse", "crps")):
                ax = axes[row, col]
                labels = []
                for model_index, model in enumerate(self.model_order):
                    y = len(self.model_order)-1-model_index
                    eligible = [c for c in cases if model in c["final_scores"]]
                    labels.append(f"{MODEL_NAMES[model]}\n{len(eligible)}/{len(cases)} valid · {len(cases)-len(eligible)} failed")
                    for domain, offset, color, marker in (("input", .12, TEAL, "o"), ("heldout", -.12, CORAL, "D")):
                        vals = np.array([c["final_scores"][model]["replication_average"][domain]["observation"][metric] for c in eligible])
                        if len(vals) == 0:
                            continue
                        jitter = np.linspace(-.035, .035, len(vals)) if len(vals)>1 else np.zeros(1)
                        ax.scatter(vals, y+offset+jitter, s=15, color=color, alpha=.28, zorder=2)
                        mean = float(vals.mean())
                        if len(vals)>1:
                            key = f"{scenario}:{model}:{domain}:{metric}".encode()
                            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
                            resamples = int(self.protocol["config"].get("bootstrap_resamples", 2000))
                            boot = vals[rng.integers(len(vals), size=(resamples, len(vals)))].mean(axis=1)
                            low, high = np.quantile(boot, [.025, .975])
                            ax.errorbar(mean, y+offset, xerr=[[max(0.,mean-low)], [max(0.,high-mean)]], fmt=marker,
                                        color=color, ms=6, lw=1.6, capsize=3, zorder=4)
                        else:
                            ax.plot(mean, y+offset, marker, color=color, ms=6)
                    if not eligible:
                        ax.text(.97, y, "No validated score", transform=ax.get_yaxis_transform(), ha="right", va="center", color=CORAL, fontsize=9)
                ax.set_yticks(np.arange(len(self.model_order))[::-1], labels if col==0 else [""]*len(self.model_order))
                ax.set_ylim(-.5, len(self.model_order)-.5)
                ax.set_title(f"{SCENARIO_NAMES.get(scenario,scenario)}  |  {metric.upper()}", loc="left", pad=10)
                ax.set_xlabel(f"Observation {metric.upper()} · smaller is better")
                ax.grid(axis="x", alpha=.8); ax.set_axisbelow(True); ax.tick_params(length=0)
                ax.spines["left"].set_visible(False)
        footer(fig, "One score per independent replication after averaging forecast origins. Failed gates remain explicit; intervals require ≥2 eligible replications and describe eligible-fit means.")
        self.save_figure(fig, "forecast_scores.png", "Final observation RMSE/CRPS at input and held-out sites, with replicate-level bootstrap uncertainty and explicit validation failures.")

    def horizon(self):
        d = self.illustration; H = len(d["mean"]); leads = np.arange(1, H+1)
        inputs = d["input_indices"].astype(int)
        heldout = np.setdiff1d(np.arange(d["mean"].shape[1]), inputs)
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True)
        fig.subplots_adjust(left=.09, right=.965, top=.84, bottom=.105, hspace=.27)
        heading(fig, "How accuracy and uncertainty evolve with lead", f"One origin, {H} uninterrupted steps · Gaussian scenario / replication 0 · {MODEL_NAMES[str(d['selected_model'])]}")
        for ids, color, label, lw in ((inputs,TEAL,"Input sites",1.8),(heldout,CORAL,"Held-out sites",1.8),
                                      (np.arange(d["mean"].shape[1]),NAVY,"All sites",2.3)):
            axes[0].plot(leads, np.sqrt(np.mean(d["error"][:, ids]**2,axis=1)),color=color,lw=lw,label=label)
            axes[1].plot(leads, d["covered90"][:, ids].mean(axis=1),color=color,lw=lw)
            axes[2].plot(leads, (d["upper90"][:, ids]-d["lower90"][:, ids]).mean(axis=1),color=color,lw=lw)
        axes[0].legend(ncol=3, loc="upper left")
        axes[0].set_ylabel("RMSE")
        axes[1].set_ylabel("90% interval coverage"); axes[1].set_ylim(0,1.04)
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1))
        axes[1].axhline(.9,color=GOLD,linestyle="--",lw=1.4)
        axes[1].text(H,.86,"Nominal 90%",ha="right",va="top",color=GOLD,fontsize=9)
        axes[2].set_ylabel("Mean 90% interval width"); axes[2].set_xlabel("Forecast lead")
        for ax in axes:
            style_axis(ax); ax.set_xlim(1,H)
        footer(fig,"Coverage is the observed proportion across sites at each lead, not an independence-based confidence interval. Main comparison scores use the protocol's separate forecast origins.")
        self.save_figure(fig,"horizon_diagnostics.png","Uninterrupted illustrative forecast: lead-wise observation RMSE, raw 90% interval coverage and width, split by input/held-out domain.")

    def trajectories(self):
        d=self.illustration; H=len(d["mean"]); leads=np.arange(1,H+1)
        anchors=np.array([[.12,.12],[.25,.85],[.72,.24]])
        sites=[int(np.argmin(np.sum((d["coordinates"]-point)**2,axis=1))) for point in anchors]
        fig,axes=plt.subplots(3,1,figsize=(12.8,9.0),sharex=True)
        fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.11,hspace=.35)
        heading(fig,"Local trajectories and predictive intervals", "Three locations chosen by fixed spatial anchors · noisy observations and raw posterior predictive bands")
        for ax,site in zip(axes,sites):
            ax.fill_between(leads,d["lower95"][:,site],d["upper95"][:,site],color=TEAL,alpha=.11,label="95% interval")
            ax.fill_between(leads,d["lower90"][:,site],d["upper90"][:,site],color=TEAL,alpha=.20,label="90% interval")
            ax.plot(leads,d["mean"][:,site],color=TEAL,lw=2,label="Forecast mean")
            ax.plot(leads,d["observation_truth"][:,site],color=NAVY,lw=1.1,marker="o",ms=2.2,label="Observed")
            x,y=d["coordinates"][site]
            status="input site" if site in d["input_indices"] else "held-out site"
            ax.set_title(f"Site {site}  ·  ({x:.2f}, {y:.2f})  ·  {status}",loc="left",fontsize=11)
            ax.set_ylabel("Response");style_axis(ax);ax.set_xlim(1,H)
        axes[0].legend(ncol=4,loc="upper right",fontsize=8.5)
        axes[-1].set_xlabel("Forecast lead")
        footer(fig,"Sites were selected by spatial location, without inspecting their forecast errors. Every band uses the same saved joint observation trajectories.")
        self.save_figure(fig,"site_trajectories.png","Observed and forecast trajectories at three prespecified spatial anchors with raw 90% and 95% predictive intervals.")

    def sampler(self):
        scenarios=self.protocol["scenarios"]
        fig,axes=plt.subplots(len(scenarios),3,figsize=(14.4,8.2),squeeze=False)
        fig.subplots_adjust(left=.07,right=.97,top=.83,bottom=.14,hspace=.50,wspace=.28)
        heading(fig,"Sampling quality is part of the result", "Every protocol-locked final fit is shown · extrema across available hyperparameter and prespecified readout-function diagnostics")
        keys=("rhat","ess_bulk","ess_tail")
        titles=("Maximum split R-hat","Minimum bulk ESS","Minimum tail ESS")
        labels=tuple(MODEL_SHORT_NAMES.get(name,name.replace("_","\n")) for name in self.model_order)
        for row,scenario in enumerate(scenarios):
            local=sorted([c for c in self.cases if c["scenario"]==scenario],key=lambda c:c["replication"])
            for col,(key,title) in enumerate(zip(keys,titles)):
                ax=axes[row,col]; missing=0;failed=0;values=[]
                for j,model in enumerate(self.model_order):
                    for r,case in enumerate(local):
                        record=case["models"].get(model,{})
                        passed=record.get("gate_passed",False);failed+=int(not passed)
                        diagnostic=record.get("diagnostics",{})
                        numbers=[]; undefined=False
                        for section in ("hyperparameters","functionals"):
                            part=diagnostic.get(section) or {}
                            for parameter in part.get("parameters",{}).values():
                                value=parameter.get(key)
                                if value is not None and np.isfinite(value):numbers.append(value)
                                else:undefined=True
                        if not numbers or undefined:
                            missing+=1;continue
                        value=max(numbers) if key=="rhat" else min(numbers)
                        values.append(value)
                        offset=(r-(len(local)-1)/2)*.055
                        ax.scatter(j+offset,value,color=TEAL if passed else CORAL,
                                   marker="o" if passed else "x",s=28,lw=1.2,alpha=.86)
                threshold=self.protocol["config"]["sampler"]["rhat_limit"] if key=="rhat" else self.protocol["config"]["sampler"]["min_ess"]
                ax.axhline(threshold,color=GOLD,ls="--",lw=1.3)
                if key!="rhat":
                    if values and min(values)>0:
                        ax.set_yscale("log")
                        ax.set_ylim(max(1,min(min(values),threshold)*.65),max(max(values),threshold)*1.5)
                    else:
                        ax.set_yscale("symlog",linthresh=1)
                        ax.set_ylim(-.05,max(800,max(values or [800])*1.5))
                else:
                    low,high=min(values+[1.]),max(values+[threshold])
                    spread=max(high-low,.005)
                    ax.set_ylim(min(.997,low-.1*spread),max(1.015,high+.35*spread))
                ax.set_xticks(range(len(self.model_order)),labels,fontsize=8.5);ax.set_xlim(-.45,len(self.model_order)-.55)
                ax.set_title(f"{SCENARIO_NAMES.get(scenario,scenario)}\n{title}",loc="left",fontsize=11)
                ax.text(.02,.97,f"Failed gates: {failed}/{len(local)*len(self.model_order)}"+(f" · undefined: {missing}" if missing else ""),
                        transform=ax.transAxes,va="top",fontsize=8.5,color=CORAL if failed or missing else GRAY)
                style_axis(ax)
        fig.legend(handles=[Line2D([],[],marker="o",color=TEAL,ls="",label="All gates passed"),
                            Line2D([],[],marker="x",color=CORAL,ls="",label="At least one gate failed"),
                            Line2D([],[],color=GOLD,ls="--",label="Prespecified threshold")],
                   loc="lower center",ncol=3,bbox_to_anchor=(.53,.07))
        footer(fig,"A small R-hat and adequate ESS are screening evidence, not proof of convergence. Failed fits remain in the ledger and are excluded from validated forecast comparisons.")
        self.save_figure(fig,"sampler_diagnostics.png","All final fits' maximum R-hat and minimum bulk/tail ESS, with composite gate outcomes and missing traces explicitly counted.")

    def sbc(self):
        result=read_json(self.source(ROOT/"results/statistical_validation.json"))
        with np.load(self.source(ROOT/"results/statistical_validation.npz"),allow_pickle=False) as z:
            ranks=z["ranks"].copy();names=z["statistic_names"].tolist()
        fig,axes=plt.subplots(2,3,figsize=(13.6,8.4))
        fig.subplots_adjust(left=.07,right=.97,top=.83,bottom=.12,hspace=.42,wspace=.25)
        heading(fig,"Calibration of joint Gaussian posterior recovery", "Conditional conjugate SBC · 200 independent prior-predictive replications · 255 joint posterior draws each")
        display=("Noise scale σ²","Readout coefficient C₀₀","Readout coefficient C₁₁","Future readout mean","Fresh future observation")
        M=result["posterior_draws_per_replicate"];N=len(ranks);bins=16
        lo,hi=stats.binom.ppf([.025,.975],N,1/bins)
        for j,(name,title) in enumerate(zip(names,display)):
            ax=axes.flat[j]
            ax.axhspan(lo,hi,color=TEAL,alpha=.08,label="Pointwise 95% binomial reference")
            ax.axhline(N/bins,color=GOLD,ls="--",lw=1.2)
            ax.hist((ranks[:,j]+.5)/(M+1),bins=bins,range=(0,1),color=NAVY,rwidth=.78,alpha=.86)
            ax.set_title(title,loc="left")
            ax.text(.98,.96,f"KS p = {result['statistics'][name]['randomized_rank_ks_p_value']:.3f}",
                    transform=ax.transAxes,ha="right",va="top",fontsize=9,color=GRAY)
            ax.set_xlim(0,1);ax.set_ylim(0,max(27,hi+4));ax.set_xlabel("Scaled posterior rank");ax.set_ylabel("Replications")
            style_axis(ax)
        ax=axes.flat[5];ax.axis("off")
        ax.text(0,.98,"WHAT THIS CHECKS",weight="bold",color=TEAL,fontsize=11,va="top")
        lines=["Fixed design, covariance and precision.","Joint coefficient / scale recovery.","", "Exact 90% interval coverage:"]
        for name,title in zip(names,("σ²","C₀₀","C₁₁","Future mean","Future observation")):
            lines.append(f"{title:20s} {result['statistics'][name]['exact_interval_coverage']:.1%}")
        lines += ["","Coverage Monte Carlo SE: 2.12 pp.","Not full recursive-ESN or hyperparameter SBC."]
        ax.text(0,.86,"\n".join(lines),va="top",linespacing=1.65,color=NAVY,fontsize=10)
        footer(fig,"Shading is a pointwise 95% binomial count reference, not a simultaneous envelope. Rank-test non-significance is not proof of correctness.")
        self.save_figure(fig,"sbc_calibration.png","Conditional conjugate SBC rank histograms, reference count band, exact interval coverages and explicit scope limits.")

    def spatial(self):
        result=read_json(self.source(ROOT/"results/spatial_approximation_validation.json"))
        self.source(ROOT/"results/spatial_approximation_validation.npz")
        fig,axes=plt.subplots(1,3,figsize=(14.4,5.4))
        fig.subplots_adjust(left=.07,right=.97,top=.74,bottom=.20,wspace=.29)
        heading(fig,"Spatial approximations: fidelity, boundaries and scale", "36 fixed sites · raw marginal variances retained · each approximation compared with a matching-smoothness Matérn")
        v=result["vecchia"];s=result["spde"]
        ax=axes[0]
        ax.plot([r["neighbors"] for r in v],[r["relative_frobenius_error_to_exact_matern"] for r in v],"o-",color=TEAL)
        ax.set_yscale("log");ax.set_xticks([4,8,16,35]);ax.set_xlabel("Predecessor neighbours");ax.set_ylabel("Relative covariance error")
        ax.set_title("Vecchia · ν = 1.5",loc="left")
        ax.annotate("All predecessors:\nexact finite-site limit",(35,v[-1]["relative_frobenius_error_to_exact_matern"]),
                    xytext=(.3,.28),textcoords="axes fraction",fontsize=8.5,color=TEAL,
                    arrowprops={"arrowstyle":"-","color":TEAL})
        style_axis(ax)
        ax=axes[1]
        primary=[r for r in s if r["comparison_purpose"]=="refinement"]
        broader=[r for r in s if r["mesh_padding"]==1.]
        ax.plot([r["mesh_resolution"] for r in primary],[r["relative_frobenius_error_to_exact_matern"] for r in primary],"o-",color=TEAL,label="Padding 0.5")
        ax.plot([r["mesh_resolution"] for r in broader],[r["relative_frobenius_error_to_exact_matern"] for r in broader],"D--",color=GOLD,label="Padding 1.0")
        pair=[r for r in s if r["comparison_purpose"]=="matched_spacing_boundary_pair"]
        ax.scatter([r["mesh_resolution"] for r in pair],[r["relative_frobenius_error_to_exact_matern"] for r in pair],s=95,facecolors="none",edgecolors=NAVY,lw=1.2,label="Equal spacing pair")
        ax.set_title("FEM SPDE · α = 2, ν = 1",loc="left");ax.set_xlabel("Mesh nodes per axis");ax.set_ylabel("Relative covariance error");ax.legend(fontsize=8.4);style_axis(ax)
        ax=axes[2]
        chosen=[v[0],v[2],s[0],s[3],s[-1]]
        labels=["Vecchia\n4","Vecchia\n16","FEM\n6 / 0.5","FEM\n26 / 0.5","FEM\n37 / 1.0"]
        artists=ax.boxplot([r["actual_spatial_diagonal"] for r in chosen],tick_labels=labels,patch_artist=True,
                          widths=.55,showfliers=True,medianprops={"color":NAVY,"linewidth":1.8},
                          whiskerprops={"color":GRAY},capprops={"color":GRAY},flierprops={"marker":".","markersize":3,"markeredgecolor":GRAY})
        for box,color in zip(artists["boxes"],[TEAL,TEAL,GOLD,GOLD,GOLD]):box.set_facecolor(color);box.set_alpha(.30);box.set_edgecolor(color)
        ax.axhline(1,color=CORAL,ls="--",lw=1.2);ax.set_ylabel("Actual spatial marginal variance")
        ax.set_title("No forced unit diagonals",loc="left");ax.tick_params(axis="x",labelsize=8);style_axis(ax)
        footer(fig,"FEM labels show resolution / padding. Sparse solves agree with their own dense approximation to 3.3e−15 relative error; fidelity to the infinite-domain Matérn is a different question.")
        self.save_figure(fig,"spatial_approximations.png","Vecchia neighbour fidelity, FEM mesh/padding sensitivity, and actual unrenormalized spatial marginal variances.")

    def animation_canvas(self, kind):
        d=self.illustration;H=len(d["mean"]);leads=np.arange(1,H+1)
        fig=plt.figure(figsize=(14.4,8.5),dpi=100)
        if kind=="forecast_comparison":
            title="Forecast and observations · evolving side by side"
            count=3;keys=("observation_truth","mean","error");titles=("Actual observations","Forecast observation mean","Signed forecast error")
        elif kind=="data_evolution":
            title="A nonlinear field in motion"
            count=2;keys=("observation_truth","generating_mean_truth");titles=("Actual observations","Simulator generating mean")
        else:
            title="Where forecast errors appear and grow"
            count=2;keys=("error","absolute_error");titles=("Signed error · forecast − observation","Absolute forecast error")
        subtitle=("Gaussian scenario · replication 0 · simulator generating mean and noisy observations · illustrative final evaluation period"
                  if kind=="data_evolution" else
                  f"Gaussian scenario · replication 0 · {MODEL_NAMES[str(d['selected_model'])]} · fixed parameter draw within every simulated trajectory")
        heading(fig,title,subtitle,top=.975)
        gs=fig.add_gridspec(3,count,left=.065,right=.96,top=.84,bottom=.13,height_ratios=[3.0,.16,1.48],wspace=.24,hspace=.24)
        axes=[fig.add_subplot(gs[0,j]) for j in range(count)]
        imgs=[]
        for j,(ax,key,label) in enumerate(zip(axes,keys,titles)):
            value=np.abs(d["error"][0]) if key=="absolute_error" else d[key][0]
            limit=None
            if kind=="data_evolution":limit=float(np.max(np.abs(np.r_[d["observation_truth"].ravel(),d["generating_mean_truth"].ravel()])))
            im=self.grid_image(ax,value,error=key=="error",absolute=key=="absolute_error",limit=limit)
            imgs.append(im);ax.set_title(label,fontsize=13,pad=9);ax.set_xlabel("")
            if j:ax.set_ylabel("")
        if kind=="forecast_comparison":
            fig.colorbar(imgs[0],cax=fig.add_subplot(gs[1,:2]),orientation="horizontal",label="Observation / forecast · fixed response scale")
            fig.colorbar(imgs[2],cax=fig.add_subplot(gs[1,2]),orientation="horizontal",label="Forecast − observation")
        elif kind=="data_evolution":
            fig.colorbar(imgs[0],cax=fig.add_subplot(gs[1,:]),orientation="horizontal",label="Response · fixed scale across time and panels")
        else:
            for j,im in enumerate(imgs):fig.colorbar(im,cax=fig.add_subplot(gs[1,j]),orientation="horizontal",label="Response error · fixed scale")
        bottom=gs[2,:].subgridspec(1,2,wspace=.28)
        left,right=fig.add_subplot(bottom[0,0]),fig.add_subplot(bottom[0,1])
        if kind=="data_evolution":
            a=d["observation_truth"].mean(axis=1);b=d["generating_mean_truth"].mean(axis=1)
            line1,=left.plot([],[],color=NAVY,label="Observed spatial mean")
            line2,=left.plot([],[],color=TEAL,label="Generating spatial mean")
            left.set_ylim(min(a.min(),b.min())-.05,max(a.max(),b.max())+.05)
            left.set_ylabel("Spatial average");left.legend(loc="upper right",fontsize=8.5)
            right.axis("off")
            textbox=right.text(.02,.92,"",transform=right.transAxes,va="top",fontsize=14,linespacing=1.7)
        else:
            a=d["running_rmse"] if kind=="forecast_comparison" else d["rmse_by_lead"]
            b=d["running_coverage"]
            line1,=left.plot([],[],color=TEAL,lw=2.3)
            line2,=right.plot([],[],color=NAVY,lw=2)
            left.set_ylim(0,max(a.max()*1.15,1e-4));left.set_ylabel("Cumulative RMSE" if kind=="forecast_comparison" else "RMSE at each lead")
            right.set_ylim(0,1.05);right.axhline(.9,color=GOLD,ls="--",lw=1.4)
            right.set_ylabel("90% interval coverage");right.yaxis.set_major_formatter(ticker.PercentFormatter(1))
            right.text(.99,.065,"Cumulative over observed leads",transform=right.transAxes,
                       ha="right",va="bottom",fontsize=8.5,color=GRAY)
            textbox=fig.text(.96,.905,"",ha="right",va="top",fontsize=12,color=TEAL,weight="bold")
        for ax in (left,right):
            if ax.axison:ax.set_xlim(1,H);ax.set_xlabel("Forecast lead" if kind!="data_evolution" else "Evaluation time step");style_axis(ax)
        foot=("Simulator generating mean is scoring-only truth and was never supplied to the fit." if kind=="data_evolution" else
              "Signed error = forecast observation mean − actual observation. All axes and color scales remain fixed across frames.")
        footer(fig,foot,y=.035)
        def update(frame):
            for im,key in zip(imgs,keys):
                value=np.abs(d["error"][frame]) if key=="absolute_error" else d[key][frame]
                im.set_data(value[d["grid_indices"]])
            line1.set_data(leads[:frame+1],a[:frame+1])
            line2.set_data(leads[:frame+1],b[:frame+1])
            if kind=="data_evolution":textbox.set_text(f"Time t = {d['times'][frame]:g}\nStep {frame+1} of {H}\nActual field SD: {d['observation_truth'][frame].std():.3f}")
            else:textbox.set_text(f"Lead {frame+1:02d} / {H}   ·   t = {d['times'][frame]:g}")
            return imgs+[line1,line2,textbox]
        return fig,update

    def animations(self):
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("ffmpeg is required for scientific animation export")
        H=len(self.illustration["mean"])
        for kind in ("data_evolution","forecast_comparison","error_evolution"):
            fig,update=self.animation_canvas(kind)
            path=self.media/f"{kind}.mp4"
            writer=animation.FFMpegWriter(fps=self.fps,codec="libx264",bitrate=-1,
                metadata={"title":kind,"comment":"Saved protocol-locked ESN observations and forecasts"},
                extra_args=["-crf","18","-pix_fmt","yuv420p","-threads","1","-movflags","+faststart"])
            with writer.saving(fig,str(path),dpi=100):
                for frame in range(H):update(frame);writer.grab_frame()
            plt.close(fig)
            print(f"video {path.name}",flush=True)
        gif=self.media/"forecast_comparison.gif"
        for width,fps,palette in ((960,self.fps,96),(800,min(self.fps,8),64)):
            filtering=f"fps={fps},scale={width}:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors={palette}:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-threads","1",
                "-i",str(self.media/"forecast_comparison.mp4"),"-filter_complex_threads","1",
                "-filter_complex",filtering,"-loop","0",str(gif)],check=True)
            if gif.stat().st_size<10_000_000:break
        if gif.stat().st_size>=10_000_000:raise RuntimeError("GIF preview exceeds the declared 10MB size budget")
        checks={}
        for name in ("data_evolution.mp4","forecast_comparison.mp4","error_evolution.mp4","forecast_comparison.gif"):
            path=self.media/name
            probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                "-show_entries","stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate,duration:format=duration,size",
                "-of","json",str(path)],text=True))
            stream=probe["streams"][0]
            if name.endswith(".mp4"):
                if (int(stream["width"]),int(stream["height"]))!=(1440,850):raise ValueError("unexpected movie dimensions")
                frames=int(stream.get("nb_read_frames",stream.get("nb_frames",0)))
                if frames!=H:raise ValueError("exported movie has wrong frame count")
                duration=float(probe["format"]["duration"])
                if abs(duration-H/self.fps)>.05:raise ValueError("exported movie has wrong duration")
                if abs(float(Fraction(stream["avg_frame_rate"]))-self.fps)>1e-6:raise ValueError("exported movie has wrong frame rate")
            checks[name]={"sha256":sha256(path),"bytes":path.stat().st_size,"ffprobe":probe,
                          "source_identity":self.illustration_identity,"protocol_sha256":self.protocol["protocol_sha256"],
                          "renderer_sha256":sha256(Path(__file__)),"source_sha256":dict(self.inputs)}
        _save_json(self.media/"media_verification.json",checks)
        for kind in ("data_evolution","forecast_comparison","error_evolution"):
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str((H//2)/self.fps),
                            "-i",str(self.media/f"{kind}.mp4"),"-frames:v","1",str(self.media/f"{kind}_middle.png")],check=True)
        contact_sheet([self.media/f"{k}_middle.png" for k in ("data_evolution","forecast_comparison","error_evolution")],
                      self.media/"media_contact_sheet.png",columns=1,thumb_width=1000)

    def manifest(self):
        required=("forecast_montage.png","forecast_scores.png","horizon_diagnostics.png",
                  "site_trajectories.png","sampler_diagnostics.png","sbc_calibration.png",
                  "spatial_approximations.png")
        missing=[name for name in required if name not in self.artifacts]
        scope=("validation_only" if self.protocol is None else
               "illustration_only" if self.summary is None else "full_study")
        value={"renderer":"report/render_figures.py","renderer_sha256":sha256(Path(__file__)),
            "created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":self.inputs,
            "protocol_sha256":None if self.protocol is None else self.protocol["protocol_sha256"],
            "render_scope":scope,
            "status":"complete" if self.summary is not None and not missing else "partial",
            "full_study_summary_loaded":self.summary is not None,
            "expected_final_cases":None if self.protocol is None else
                len(self.protocol["scenarios"])*int(self.protocol["replications"]),
            "loaded_final_cases":len(self.cases),"missing_required_figures":missing,
            "illustration":getattr(self,"illustration_identity",None),"figures":self.artifacts,"warnings":self.warnings,
            "style":{"font":"DejaVu Sans","navy":NAVY,"teal":TEAL,"coral":CORAL,"gold":GOLD},
            "uncertainty_policy":"Forecast score intervals resample independent replication averages; no forecast-origin pseudoreplication.",
            "score_uncertainty":{"resamples":None if self.protocol is None else self.protocol["config"].get("bootstrap_resamples",2000),
                "interval":"95% percentile bootstrap of eligible-fit score means; paired contrast claims use the saved study summary"},
            "selection_policy":"No fitting or model selection performed by this renderer."}
        _save_json(self.figures/"figure_manifest.json",value)
        contact_sheet([self.figures/name for name in self.artifacts],self.figures/"figure_contact_sheet.png")


def contact_sheet(paths, output, columns=2, thumb_width=700):
    if not paths:return
    font=ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"),17)
    prepared=[]
    for path in paths:
        with Image.open(path) as raw:
            im=raw.convert("RGB");height=round(im.height*thumb_width/im.width)
            im=im.resize((thumb_width,height),Image.Resampling.LANCZOS)
            tile=Image.new("RGB",(thumb_width+24,height+45),"white")
            tile.paste(im,(12,33));ImageDraw.Draw(tile).text((12,7),Path(path).name,fill=NAVY,font=font)
            prepared.append(tile)
    cell_width=max(im.width for im in prepared);cell_height=max(im.height for im in prepared)
    rows=(len(prepared)+columns-1)//columns
    sheet=Image.new("RGB",(columns*cell_width,rows*cell_height),"#e9eef1")
    for i,im in enumerate(prepared):sheet.paste(im,((i%columns)*cell_width,(i//columns)*cell_height))
    sheet.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"report/controlled_study")
    parser.add_argument("--output",type=Path,default=ROOT/"report")
    parser.add_argument("--fps",type=int,default=10)
    parser.add_argument("--validation-only",action="store_true")
    parser.add_argument("--illustration-only",action="store_true",help="Render the prespecified completed case without waiting for other cases; no aggregate comparisons")
    parser.add_argument("--skip-media",action="store_true")
    args=parser.parse_args()
    renderer=Renderer(args.study.resolve(),args.output.resolve(),args.fps)
    renderer.sbc();renderer.spatial()
    if not args.validation_only:
        renderer.load_study(illustration_only=args.illustration_only)
        if not args.illustration_only:renderer.scores();renderer.sampler()
        if renderer.illustration is not None:
            renderer.montage();renderer.horizon();renderer.trajectories()
            if not args.skip_media:renderer.animations()
    renderer.manifest()
    print(json.dumps({"figures":list(renderer.artifacts),"warnings":renderer.warnings,
                      "manifest":str(renderer.figures/"figure_manifest.json")},indent=2),flush=True)


if __name__=="__main__":main()
