#!/usr/bin/env python3
"""Render a receipt-bound KS improvement report; no fitting or data access.

The input is the compact report directory from build_ks_improvement_report.py.
Every member of its artifact manifest is verified before rendering. This
renderer adds explanation and layout, never recomputes scientific scores.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image, KeepTogether, Preformatted)
from pypdf import PdfReader


TRACKS = {"A": "Interleaved unseen-site forecasting", "B": "All-observed temporal forecasting",
          "C": "Contiguous unseen-site forecasting"}
MODEL_NAMES = {"training_mean": "Training mean", "persistence": "Persistence + periodic decoder",
               "linear_var": "Fixed linear/POD VAR", "ridge_esn": "Fixed sparse ridge ESN",
               "pod_nvar": "Fixed POD-NVAR"}
NAVY, TEAL, GRAY = map(colors.HexColor, ("#142d40", "#116a70", "#52616b"))
LIGHT, ORANGE = map(colors.HexColor, ("#eef3f5", "#a65018"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=4):
    if value is None:
        return "Pending / undefined"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite report summary")
    if value and (abs(value) >= 10000 or abs(value) < 10 ** (-digits)):
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def interval(stat, digits=4):
    if stat.get("mean") is None:
        return "Undefined"
    ci = stat.get("ci95")
    return fmt(stat["mean"], digits) + (f" [{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]" if ci else "")


def role(row):
    if "cheap_anchor" in row.get("roles", []):
        return MODEL_NAMES.get(row["candidate_id"], row["candidate_id"])
    return " / ".join({"main": "Broader search, fixed prior", "anchor": "Original selected configuration",
                       "sensitivity": "Prior sensitivity"}.get(x, x) for x in row.get("roles", []))


class Renderer:
    def __init__(self, source, output, revision):
        self.source, self.output, self.revision = Path(source), Path(output), revision
        self.manifest = load(self.source / "artifact_manifest.json")
        for name, expected in self.manifest["files"].items():
            p = self.source / name
            if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
                raise ValueError(f"Input report artifact integrity failure: {name}")
        self.r = load(self.source / "report.json")
        self.freeze = load(self.source / "selected_configurations.json")
        self.registry = load(self.source / "candidate_registry.json")
        self.dynamics = load(self.source / "realized_dynamics.json")
        self.seed = load(self.source / "seed_sensitivity.json")
        self.fits = load(self.source / "fit_diagnostics.json")
        source_manifest = load(self.source / "source_manifest.json")
        diagnosis_path = Path(__file__).resolve().parents[1] / "docs/applications/evidence/ks_improvement_initial_diagnosis.json"
        self.diagnosis = None
        if source_manifest.get("initial_diagnosis_sha256"):
            if sha(diagnosis_path) != source_manifest["initial_diagnosis_sha256"]:
                raise ValueError("Initial diagnostic evidence changed after the compact report")
            self.diagnosis = load(diagnosis_path)
        fonts = Path("/usr/share/fonts/truetype/dejavu")
        for name, file in (("KS", "DejaVuSans.ttf"), ("KS-Bold", "DejaVuSans-Bold.ttf"),
                           ("KS-Mono", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
        pdfmetrics.registerFontFamily("KS", normal="KS", bold="KS-Bold", italic="KS", boldItalic="KS-Bold")
        base = dict(fontName="KS", textColor=NAVY, spaceAfter=7, allowWidows=0, allowOrphans=0)
        self.styles = {
            "body": ParagraphStyle("body", fontSize=9.2, leading=13.4, **base),
            "small": ParagraphStyle("small", fontSize=7.7, leading=10.5, **base),
            "title": ParagraphStyle("title", fontSize=26, leading=31, fontName="KS-Bold", textColor=NAVY, spaceAfter=17),
            "h1": ParagraphStyle("h1", fontSize=16, leading=20, fontName="KS-Bold", textColor=TEAL, spaceAfter=12, keepWithNext=True),
            "h2": ParagraphStyle("h2", fontSize=11.2, leading=15, fontName="KS-Bold", textColor=NAVY, spaceBefore=8, spaceAfter=6, keepWithNext=True),
            "cell": ParagraphStyle("cell", fontSize=7.6, leading=10.4, fontName="KS", textColor=NAVY, splitLongWords=True),
            "head": ParagraphStyle("head", fontSize=7.5, leading=10.1, fontName="KS-Bold", textColor=colors.white),
        }
        self.story = []
        self.width = A4[0] - 80

    def p(self, text, style="body"):
        # Only deliberate lightweight ReportLab markup is permitted by callers.
        self.story.append(Paragraph(text.replace("–", "-").replace("—", "-"), self.styles[style]))

    def heading(self, title, new_page=True):
        if new_page and self.story:
            self.story.append(PageBreak())
        self.p(html.escape(title), "h1")

    def table(self, headings, rows, weights=None):
        weights = weights or [1] * len(headings)
        widths = [self.width * w / sum(weights) for w in weights]
        def cell(x, head=False):
            value = str(x).replace("–", "-").replace("—", "-")
            return Paragraph(html.escape(value).replace("\n", "<br/>"), self.styles["head" if head else "cell"])
        data = [[cell(v, True) for v in headings]] + [[cell(v) for v in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY),
                              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                              ("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 6),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                              ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                              ("LINEBELOW", (0, 0), (-1, 0), .6, TEAL)]))
        self.story.extend([t, Spacer(1, 10)])

    def figure(self, name, maximum_height=540, caption=None):
        if name not in self.r["figures"]:
            self.p("This figure is pending because its required evidence is not complete.", "small")
            return
        p = self.source / name
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            w, h = im.size
        scale = min(self.width / w, maximum_height / h)
        self.story.append(Image(str(p), width=w*scale, height=h*scale, hAlign="CENTER"))
        if caption:
            self.p(caption, "small")

    def introduction(self):
        self.p("KURAMOTO-SIVASHINSKY", "small")
        self.p("A broader Bayesian\nST-ESN improvement study".replace("\n", "<br/>"), "title")
        self.p("Custom, literature-grounded experiment | validation-only follow-up", "h2")
        self.p(f"Evidence status: <b>{self.r['status'].upper()}</b>. Generated from immutable receipts dated "
               f"{html.escape(self.r['created_utc'])}.")
        if self.r["status"] != "complete":
            self.p("<b>INCOMPLETE PREVIEW.</b> This document does not establish completion of screening, selection, "
                   "confirmation or sensitivity. Pending schedules remain visible. No unfinished score is imputed.")
        self.p("The original KS experiment used four bundled Bayesian candidates per main spatial track. This follow-up "
               "evaluates 104 explicit configurations: 40 for interleaved unseen sites, 40 for all observed sites, and 24 "
               "for an independently selected contiguous-site experiment. The main search preserves the accepted prior "
               "and likelihood. Prior-scale sensitivity is a separate stratum.")
        self.p("The original test results had already been inspected before this improvement campaign. All new selection "
               "and confirmation use validation intervals; 14 trajectories are excluded from the new selection. This "
               "is an exploratory improvement study, not a fresh untouched test or an exact published leaderboard reproduction.")
        rows = []
        for track in self.r["tracks"]:
            cohorts = [r for r in self.r["confirmation"] if r["track"] == track]
            old = next((r for r in cohorts if "anchor" in r["roles"]), None)
            new = next((r for r in cohorts if "main" in r["roles"]), None)
            if old and new:
                rows.append([track, f"{old['complete_trajectories']}/14 → {new['complete_trajectories']}/14",
                             fmt(old["metrics"]["rmse"]["mean"]), fmt(new["metrics"]["rmse"]["mean"]),
                             fmt(old["metrics"]["crps"]["mean"]), fmt(new["metrics"]["crps"]["mean"])])
        if rows:
            self.table(["Track", "Complete old → new", "Old RMSE", "New RMSE", "Old CRPS", "New CRPS"], rows,
                       [.6, 1.6, 1, 1, 1, 1])
            self.p("These are complete-trajectory summaries. See the reliability census and paired trajectory intervals "
                   "before interpreting a difference. A completed run may include inference failures; completion does "
                   "not make every probabilistic result eligible.", "small")
        self.p("The best evaluated configuration is conditional on the finite roster, selection objective, model seed "
               "and data protocol. This report makes no claim of universal optimality or state of the art.")
        self.p(f"Source revision: {html.escape(self.revision)}<br/>Experiment: {self.r['experiment_identity']}<br/>"
               f"Frozen selection: {self.r['freeze_identity'] or 'pending'}", "small")

    def contract(self):
        self.heading("1. Scientific contract and preserved model")
        self.p("The delivered data solve u_t = -u u_x - u_xx - u_xxxx on the periodic domain [0,22), with "
               "128 solver points and a 64-site observation panel. Saved cadence is 0.25 physical time units; "
               "100 forecast steps span 25 units. Only the declared observation-panel inputs enter the models.")
        self.table(["Boundary", "Executed policy"], [
            ["Training / validation / original test", "[0,6000) / [6000,8000) / [8000,10000); zero-based, half-open. No new test scoring."],
            ["A: interleaved", "39 training sites; 13 validation sites; 12 original test sites. Only training-site histories initialize forecasts."],
            ["B: all observed", "All 64 permitted sites for training/history and validation forecasts. Separately selected configuration."],
            ["C: contiguous", "40 training / 12 validation / 12 test sites. New independent validation selection; not transfer of A's winner."],
            ["History and horizon", f"Common history {self.r['history']}; autonomous horizon 100. History is y[a-history:a], targets y[a:a+100]."],
            ["Transform / learned features", "Delivered trajectory/protocol-specific training-only scalar transform. PCA, feature scales and basis statistics obey permitted fitting boundaries."],
            ["Confirmation", "14 trajectories × 20 nonoverlapping validation origins, 6000,6100,...,7900; 512 complete paths per Bayesian origin."],
        ], [1.3, 3.3])
        self.p("Accepted model", "h2")
        self.p("On the delivered standardized scale, y_t = Φ C z_t + e_t, with e_t independently distributed over "
               "time as Normal(0, σ²[I + q K_ell]). K_ell is a Matérn-3/2 covariance on the valid unit-circle "
               "embedding of physical coordinates. The spatial basis evaluates never-trained coordinates from "
               "static geometry; hidden response histories are excluded.")
        self.p("The readout prior is vec(C) | σ², λ ~ Normal(0, σ² λ⁻¹ I), including the intercept. The accepted "
               "hierarchy uses σ² ~ inverse-gamma(2,0.05), log λ ~ Normal(log(0.1),1.5²), log q ~ Normal(0,1.25²), "
               "and a lognormal range hierarchy with standard deviation 0.65. The track-specific original selected "
               "range-prior median is preserved in the main stratum; changes are explicitly labeled sensitivity.")
        self.p("Fixed prior means that these coefficient, noise and covariance hyperpriors remain unchanged conditional "
               "on each architecture. Changing a spatial basis or its dimension changes the induced prior over fields; "
               "it does not make two architectures the same stochastic process.", "small")
        self.p("Native PCA-ESN features combine lagged permitted-site scores, sparse leaky tanh reservoirs and declared "
               "squared-state terms. Inference uses the existing exact collapsed random-walk Metropolis target with "
               "conditional Gaussian/inverse-gamma parameter recovery. It is not HMC or a deterministic substitute.")
        self.p("The reservoir receives raw conditioning-prefix whitened PCA scores. The bounded map 3 tanh(a/3) "
               "applies only to score terms in the readout. Recurrent weights are sparse Gaussian draws scaled to "
               "the declared operator norm; input-weight standard deviation is input_scale divided by the square "
               "root of the incoming dimension, and fixed biases are uniform on [-0.3,0.3]. Readout feature centering "
               "and scaling use the final two thirds of the conditioning prefix, with scale floor 0.04. The spectral "
               "radius is measured separately and must not be confused with the configured operator norm.", "small")
        self.p("Each predictive path holds its static parameter draw fixed, generates the accepted spatially correlated "
               "innovations, and feeds back its own generated input-site field. Primary point forecasts average complete "
               "paths. Residual uncertainty remains part of this fitted clean-field predictive model; it is not an "
               "arbitrary added measurement-noise floor.")

    def search(self):
        self.heading("2. Search breadth, efficiency and qualification")
        self.table(["Dimension", "Evaluated opportunity"], [
            ["Reservoir capacity", "32, 64, 128, 256 and 512 units across the A/B roster; smaller dedicated C proposals."],
            ["Recurrence", "Operator norms 0.3, 0.6, 0.9, 0.98; scalar and mixed leaks, including 0.05, 0.1, 0.2, 0.5, 1."],
            ["Input and sparsity", "Input scales 0.05–2; connectivity 0.02, 0.08, 0.2. Realized eigenvalue radii are recorded separately."],
            ["Temporal features", "PCA dimensions 8–32; lags [1], [1,2], [1,2,4]; squared terms including zero; conditioning prefixes through 2048, retaining legacy settings."],
            ["Spatial features", "8, 12, 20, 32 columns; widths 0.15–1, including the legacy width 0.4. Track C emphasizes geometry stability."],
            ["Prior sensitivity", "Readout precision-prior median ×0.1 or ×10; range-prior median ×0.5 or ×2. Never pooled into fixed-prior selection."],
        ], [1.25, 3.75])
        self.p("These ranges describe the generated non-Cartesian roster, not every combination. The registry hashes "
               "effective parameters and deduplicates aliases. Controlled single-factor contrasts and seeded joint "
               "proposals are retained separately. All four original candidates remain references.")
        self.table(["Stage", "Schedule and rule"], [
            ["Training-only preflight", "All 104 candidates on trajectories 0, 5. All 208 checks passed at history 512; the common history was fixed before screening."],
            ["Initial screen", "Two trajectories, four validation origins, full permitted training panel, 256 paths. One common seed 20261105."],
            ["Promotion", "Four main candidates plus original anchor and one separate prior sensitivity; six trajectories, eight origins, 512 paths."],
            ["Precision and freeze", "Four repeated 512-path ensembles on two fixed cases for each eligible promoted main candidate. Candidate-specific near-tie tolerance; lower readout dimension breaks near ties."],
            ["Confirmation", "Freeze first; evaluate on the other 14 trajectories, all20 validation origins. No reselection on these outcomes."],
            ["Model-seed sensitivity", "Additional seeds 20261106, 20261107 on trajectories 0, 5, 10, 15 and eight origins; matched main-seed references."],
        ], [1.25, 3.75])
        self.p("Selection minimizes episode-balanced CRPS, with equal weights for the mean losses in leads 1-10, 11-25, "
               "26-50 and 51-100. Reported ordinary 100-step CRPS and RMSE are distinct summaries. MCMC keeps four chains, "
               "1000 warmup and 2000 retained states per chain, with one bounded 2000-draw continuation. Qualification "
               "requires rank-normalized split R-hat below 1.01 and bulk/tail ESS ≥ 400 for monitored quantities [3].")
        self.p("All sampled hyperstates are checked. All readout coefficients and ten scale/readout/predictive functionals "
               "are checked using 512 evenly spaced retained states per chain. Predictive precision repetitions measure "
               "conditional simulation variability; they are not substitutes for chain convergence or trajectory-level uncertainty.")

    def result_pages(self):
        for track in self.r["tracks"]:
            rows = [r for r in self.r["confirmation"] if r["track"] == track]
            self.heading(f"3{track}. {TRACKS[track]}")
            if not rows:
                self.p("Confirmation is pending. No performance conclusion is reported.")
                continue
            self.table(["Method / role", "Complete trajectories", "RMSE", "MAE", "CRPS", "90% coverage / width"], [
                [role(r), f"{r['complete_trajectories']}/14", fmt(r["metrics"]["rmse"]["mean"]),
                 fmt(r["metrics"]["mae"]["mean"]),
                 ("Point only" if "cheap_anchor" in r.get("roles", [])
                  else fmt(r["metrics"]["crps"]["mean"])),
                 (fmt(100*r["metrics"]["coverage90"]["mean"],1)+"% / "+fmt(r["metrics"]["width90"]["mean"]))
                 if r["metrics"]["coverage90"]["mean"] is not None else "Not defined"] for r in rows], [2.25,1,1,1,1,1.4])
            self.table(["Role", "Finite /280", "Rollout failed", "Inference blocked", "Pending / other blocked", "Finite divergent"], [
                [role(r), r["finite_successful_origins"], r["failed_rollout_origins"], r["unqualified_blocked_origins"],
                 f"{r['pending_origins']} / {r['failed_job_blocked_origins']}", r["finite_divergent_origins"]] for r in rows], [2.2,1,1,1,1.4,1])
            contrasts = [x for x in self.r["paired_comparisons"] if x["track"] == track and
                         x["purpose"] == "preserved_prior_vs_old" and x["metric"] in ("rmse", "crps")]
            for c in contrasts:
                self.p(f"<b>New minus original {c['metric'].upper()}:</b> {interval(c)}; paired trajectories n={c['n']}. "
                       "Negative differences favor the new selection. Intervals are conditional on complete pairs and selected configurations.", "small")
            self.figure(f"horizon_{track}.png", maximum_height=350,
                        caption="Trajectory-first curves with pointwise 95% trajectory-bootstrap intervals. Origins, cells, chains and paths are not independent scientific replicates. All finite extremes remain in the figures; a labeled nonlinear scale may be used.")
            self.p("RMSE pools squared errors within each trajectory before taking a square root, then weights trajectories "
                   "equally. Successful-only accuracy excludes an entire trajectory if any scheduled origin is unavailable; "
                   "the census above retains those missing outcomes. A method with fewer complete trajectories cannot be "
                   "declared generally superior from its successful-only mean.", "small")

    def parameter_pages(self):
        if not self.freeze:
            return
        for track, selection in self.freeze["selected"].items():
            self.heading(f"4{track}. Exact Bayesian settings")
            candidates = {c["id"]: c for c in self.registry["tracks"][track]["candidates"]}
            ids = [selection["anchor_id"], selection["main"]["id"]]
            configs = [candidates[cid]["config"] for cid in ids]
            def values(c):
                f, s = c["features"], c["spatial_mean"]
                radial = len(s["centers"]) if s.get("centers") is not None else s["side"]**2
                p = 1 + (2 if s.get("coordinate_terms") else 0) + radial
                d = 1 + len(f["lags"])*f["modes"] + sum(f["units"]) + f["squares"] + f["products"]
                return {"Reservoir units": f["units"], "Operator norm": f["recurrent_norm"],
                        "Leak pattern": f["leak_patterns"], "Input scale": f["input_scales"],
                        "Connectivity": f["connectivity"], "PCA modes": f["modes"], "Lags": f["lags"],
                        "Squared terms": f["squares"], "Conditioning rows": c["conditioning"],
                        "Usable likelihood targets": 6000-c["conditioning"], "Spatial columns": p,
                        "Spatial width": s["width"], "Temporal columns": d, "Readout coefficients": p*d,
                        "Range-prior median": f"{math.exp(c['covariance']['log_ell_mean']):.6g}",
                        "Precision-prior median": f"{math.exp(c['prior']['mean_log_precision']):.6g}"}
            old, new = map(values, configs)
            self.table(["Effective parameter", "Original selected", "New fixed-prior selected"],
                       [[k, old[k], new[k]] for k in old], [1.65,1.35,1.5])
            self.p("Original ID: "+html.escape(ids[0])+"<br/>New ID: "+html.escape(ids[1]), "small")
            self.p("Each range and precision prior in this main comparison remains the track's accepted original selected "
                   "hyperprior. More conditioning rows reduce usable likelihood targets; they are not the origin-history "
                   "length. All compared origins use the common 512-state history.")
            observed = []
            for cid in dict.fromkeys(ids):
                dynamics = next((d for d in self.dynamics if d["track"] == track and d["candidate_id"] == cid), None)
                if dynamics:
                    values = [x["spectral_radius"] for r in dynamics["observations"] for x in r["layers"]]
                    conditions = [r["geometry_condition"] for r in dynamics["observations"]]
                    observed.append([cid, f"{min(values):.6f}–{max(values):.6f}", f"{min(conditions):.4g}–{max(conditions):.4g}"])
            if observed:
                self.table(["Configuration", "Realized eigenvalue radius", "Training spatial condition number"], observed, [2,1.3,1.7])
            self.p("The native recurrent matrix is normalized by operator norm, not eigenvalue radius. Its admissible "
                   "operator-norm sweep does not reproduce a conventional ESN spectral-radius sweep. Geometry "
                   "conditioning uses known coordinates; it does not reveal withheld response histories.", "small")
            if selection.get("sensitivity"):
                self.p("Separate prior sensitivity: "+html.escape(selection["sensitivity"]["id"])+". Its full effective "
                       "configuration is retained in selected_configurations.json. It is not relabeled as the unchanged main prior.", "small")

    def sensitivity_pages(self):
        self.heading("5. Model-seed sensitivity and diagnostic interpretation")
        rows = self.seed.get("rows", [])
        if rows:
            self.table(["Track", "Role", "Seed", "Complete /4", "RMSE", "CRPS"], [
                [r["track"], role(r), r["seed"], r["complete_trajectories"],
                 fmt(r["metrics"]["rmse"]["mean"]), fmt(r["metrics"]["crps"]["mean"])] for r in rows], [.55,2.25,1.15,.8,1,1])
        else:
            self.p("Model-seed sensitivity is pending.")
        self.p("The main seed is evaluated on the same four trajectories and eight origins as each additional seed. "
               "These n ≤ 4 comparisons are conditional sensitivity analyses, not comparisons with the 14-trajectory "
               "confirmation mean. Model seeds are not posterior samples, and their predictions are not pooled into "
               "an invented Bayesian model-averaging distribution.")
        if self.diagnosis:
            self.heading("5b. What the retained-fit diagnostic establishes")
            cases = sorted(self.diagnosis["cases"], key=lambda c: c["first_target"])
            labels = {"full_predictive": "Full posterior predictive mean",
                      "parameter_only": "Parameter draws, zero future innovations",
                      "fixed_parameter_innovations": "Fixed parameters, fitted innovations",
                      "deterministic_conditional_mean": "Deterministic conditional readout",
                      "teacher_forced_one_step": "Teacher-forced one-step diagnostic"}
            self.table(["Regime"] + [f"RMSE: target {c['first_target']}" for c in cases],
                       [[label] + [fmt(c["modes"][mode]["scores"]["rmse"]) for c in cases]
                        for mode, label in labels.items()], [2.4, 1.3, 1.3])
            self.p("The original A/trajectory 0 fit has small teacher-forced one-step errors, but much larger autonomous "
                   "errors. Parameter-only forecasts do not consistently improve the full predictive score. These "
                   "are two correlated validation cases, and the interventions are not additive variance components. "
                   "They motivate examining autonomous feedback; they do not identify one unique cause of poor forecasting.")
            self.p("The deterministic readout is the conditional coefficient mean at the declared componentwise median "
                   "log-hyperparameters. It is neither the marginal posterior mean coefficient matrix nor the posterior "
                   "predictive mean. Teacher forcing supplies the permitted input-site field between one-step predictions "
                   "and is explicitly nonautonomous. Neither diagnostic belongs in the native probabilistic ranking.")
            self.p("Only the full posterior predictive regime retains every accepted uncertainty component. No variance "
                   "floor was added and no innovation term was removed from the actual improvement campaign. The "
                   "diagnostic source is checksum-bound by source_manifest.json.", "small")

    def figure_pages(self):
        for track in self.r["tracks"]:
            for name, title, caption in [
                (f"search_{track}.png", f"6{track}. Search evidence", "All evaluated configurations retain their eligibility and failure counts in search_candidates.csv. Legacy prior references, fixed-prior candidates and prior sensitivity remain separate. These scores are selection data."),
                (f"compute_{track}.png", f"7{track}. Accuracy and compute", "Fitting cost is measured wall time per fitted model. Forecast and search time are reported separately; neither candidate count nor epoch count alone measures comparable opportunity."),
                (f"example_{track}_t01_a6000.png", f"8{track}. Predeclared space-time example", "Trajectory 1, first target 6000 was fixed before confirmation. Only official validation target sites are scored. The site map distinguishes permitted inputs from held-out targets; categorical target rows do not imply interpolation of a fully observed field."),
                (f"intervals_{track}_t01_a6000.png", f"9{track}. Predictive intervals at fixed coordinates", "Coordinates follow the first/middle/last official target-site rule. These are marginal 90% clean-field predictive intervals. They do not describe simultaneous field or path coverage."),
            ]:
                if name in self.r["figures"]:
                    self.heading(title)
                    self.figure(name, 605, caption)

    def operations(self):
        self.heading("10. Computation, acceptance and limitations")
        ops = self.r["operations"]
        self.table(["Stage", "Worker wall hours"], [[k, fmt(v/3600,3)] for k,v in ops["attempt_worker_seconds_by_stage"].items()] +
                   [["Recorded preparation", fmt(ops["preparation_worker_seconds"]/3600,3)],
                    ["Cumulative attempt + preparation", fmt((ops["attempt_worker_seconds"]+ops["preparation_worker_seconds"])/3600,3)]], [3,2])
        self.p("The environment has an 8-core CPU quota and an 8 GiB container memory limit; no GPU was used. Four model "
               "workers run with one numerical thread each. The measured 512-unit pilot required 64.82 s for a qualified fit, "
               "5.53 s for one 256-path forecast and 537,538,560 bytes peak RSS. Derived matrices use the accepted separable "
               "structure; the full space-time tensor design and a giant readout covariance are not materialized.")
        self.p("The configured admission thresholds are 12 worker hours overall, 6 worker hours for search, and 4 elapsed "
               "hours; six worker hours are reserved for confirmation and diagnostics. Already admitted jobs may finish "
               "after a threshold. At four workers with 1800-second per-job timeouts, this creates a bounded in-flight "
               "overrun allowance. Actual recorded use, not the nominal threshold alone, determines budget compliance.")
        self.p("Scheduling also requires the 1.5× worker-memory estimate to fit the combined allowance and 65% of "
               "container-aware headroom. The disk reserve is 12,565,873,460 bytes. Cache advice releases pages of closed "
               "fit files after proved child termination; it neither deletes checkpoints nor relaxes admission.")
        self.p("An early mutable-ledger inconsistency was repaired by verifying 208 immutable preflight artifacts, "
               "carrying their identities and prior time forward, and adopting immutable terminal receipts as authority. "
               "The responsible external mechanism was not established. No scientific candidate, prior, data, forecast "
               "or score was changed by that recovery. The original run remains evidence.")
        acceptance = self.source / "selected_invariance.json"
        if acceptance.exists():
            a = load(acceptance)
            self.p(f"<b>Selected-artifact acceptance:</b> {a['passed_cases']}/{a['expected_cases']} cases passed; "
                   f"failed={a['failed_cases']}, pending={len(a['pending'])}. Tests fix fitted parameters and randomness, "
                   "perturb hidden histories and future labels, and separately refit preprocessing/features on unchanged "
                   "permitted training inputs. They establish output separation, not future-free physical disk access.")
        self.p("The finite search can leave important settings at a search boundary. Conditional confirmation can "
               "support improvement within this experiment, but cannot establish a globally optimal architecture. "
               "A larger capacity, a changed reservoir normalization or a different periodic basis would be a separately "
               "identified extension. The ordinary CNN/FNO, GP and GRU are not silently retuned or replaced in this native-only follow-up.")

    def references(self):
        self.heading("11. Reproduction and evidence map")
        self.p("The repository contains the full frozen 104-candidate registry, source-bound runner, targeted checks, "
               "completed-stage selection records, exact selected configurations, compact episode/horizon scores and "
               "the report builder. requirements-ks-improvement.lock records the compatible native-only environment.")
        self.table(["Evidence", "Purpose"], [
            ["report.json / confirmation.csv", "Final census, trajectory-first scores, reliability and cost."],
            ["selected_configurations.json", "Frozen IDs, exact effective parameters, precision/tie rule, confirmation schedule."],
            ["search_candidates.csv / controlled_screen_contrasts.csv", "Every scored stage candidate and matched single-factor contrasts. Selection diagnostics, n=2 at initial screen."],
            ["episode_results.json / horizon_curves.json", "Complete trajectory summaries and trajectory-bootstrap horizon curves."],
            ["fit_diagnostics.json / seed_sensitivity.json", "Inference qualification, monitored extrema, continuations and matched seed effects."],
            ["source_manifest.json / artifact_manifest.json", "Hashes and sizes for source receipts and generated compact artifacts."],
            ["ks_improvement_reproduction.md", "Environment, retrieval, full execution, stage commands, resume semantics and historical-budget handling."],
        ], [2.2,2.8])
        self.p("Typical report rebuild, from the repository:", "h2")
        continuation = chr(92)
        command = "\n".join([
            "export PYTHONPATH=src",
            "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
            f"python tools/build_ks_improvement_report.py {continuation}",
            "  --results RUN_ROOT --output REPORT_ROOT",
            f"python tools/build_ks_improvement_pdf.py {continuation}",
            f"  --report REPORT_ROOT --output ks_improvement_report.pdf {continuation}",
            "  --revision PUBLISHED_REVISION",
        ])
        self.story.append(Preformatted(command, ParagraphStyle("code", fontName="KS-Mono", fontSize=7.4, leading=10.5, textColor=NAVY), maxLineLength=90))
        self.p("Primary sources", "h2")
        refs = [
            ("1", "Pathak et al. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data.", "https://doi.org/10.1103/PhysRevLett.120.024102"),
            ("2", "Vlachas et al. (2020). Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for Forecasting Complex Spatiotemporal Dynamics.", "https://arxiv.org/abs/1910.05266"),
            ("3", "Vehtari et al. (2021). Rank-normalization, folding, and localization: improved R-hat.", "https://arxiv.org/abs/1903.08008"),
            ("4", "Gneiting and Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.", "https://doi.org/10.1198/016214506000001437"),
            ("5", "Li et al. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.", "https://www.jmlr.org/papers/v18/16-558.html"),
            ("6", "Bergstra and Bengio (2012). Random Search for Hyper-Parameter Optimization.", "https://www.jmlr.org/papers/v13/bergstra12a.html"),
            ("7", "Gauthier et al. (2021). Next generation reservoir computing.", "https://arxiv.org/abs/2106.07688"),
        ]
        for number, title, url in refs:
            self.p(f"[{number}] {html.escape(title)} <link href='{url}' color='#116a70'>{html.escape(url)}</link>", "small")
        self.p("These papers motivate architecture and evaluation choices. The prepared dataset and this frozen protocol "
               "govern the experiment. The staged search is an adaptation, not an implementation of every Hyperband guarantee.", "small")

    def footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cfdae0"))
        canvas.line(40, 39, A4[0]-40, 39)
        canvas.setFont("KS", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(40, 26, f"KS improvement | {self.r['status']} | exploratory validation | {self.r['experiment_identity'][:12]}")
        canvas.drawRightString(A4[0]-40, 26, str(doc.page))
        canvas.restoreState()

    def build(self):
        self.introduction(); self.contract(); self.search(); self.result_pages()
        self.parameter_pages(); self.sensitivity_pages(); self.figure_pages(); self.operations(); self.references()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(".tmp.pdf")
        doc = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=40, leftMargin=40,
                                topMargin=42, bottomMargin=52, title="Bayesian KS improvement study",
                                author="bayes-stesn research workspace", pageCompression=1)
        doc.build(self.story, onFirstPage=self.footer, onLaterPages=self.footer)
        reader = PdfReader(temporary)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(reader.pages) < 5 or "Scientific contract" not in text or "Reproduction" not in text:
            raise ValueError("PDF structural text verification failed")
        temporary.replace(self.output)
        receipt = {"schema": "ks-improvement-pdf/1", "status": self.r["status"],
                   "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "experiment_identity": self.r["experiment_identity"], "freeze_identity": self.r["freeze_identity"],
                   "report_manifest_sha256": sha(self.source/"artifact_manifest.json"), "source_revision": self.revision,
                   "renderer_sha256": sha(__file__), "pdf_bytes": self.output.stat().st_size,
                   "pdf_sha256": sha(self.output), "pages": len(reader.pages), "new_fits": 0,
                   "dataset_read": False, "scores_recomputed": False, "visual_qa": "pending"}
        self.output.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    Renderer(a.report, a.output, a.revision).build()
