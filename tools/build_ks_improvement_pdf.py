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


LEADS = (1, 5, 10, 25, 50, 100)
DIAGNOSTIC_BLOCKS = ("hyperparameters", "coefficients", "functionals")


def percent_interval(stat):
    """Format an already aggregated proportion; never estimate a new interval."""
    return interval({"mean": None if stat.get("mean") is None else 100 * stat["mean"],
                     "ci95": None if stat.get("ci95") is None else [100 * x for x in stat["ci95"]]}, 1)


def diagnostic_summary(rows, block=None):
    """Summarize compact fit receipts only, counting reused fits once."""
    unique = {}
    for row in rows:
        key = row["fit_key"]
        if key in unique and unique[key] != row:
            raise ValueError("Conflicting compact diagnostics for the same fitted artifact")
        unique[key] = row
    diagnostics = [r.get("diagnostics", {}) for r in unique.values()]
    names = (block,) if block else DIAGNOSTIC_BLOCKS
    blocks = [d.get("blocks", {}).get(name, {}) for d in diagnostics for name in names]

    def finite_values(key):
        return [float(b[key]) for b in blocks
                if b.get(key) is not None and math.isfinite(float(b[key]))]

    def extreme(key, operation):
        values = finite_values(key)
        return operation(values) if values else None

    continuations = [d["continuations"] for d in diagnostics if d.get("continuations") is not None]
    checked = [b["monitored"] for b in blocks if b.get("monitored") is not None]
    failed = [b["failed_quantities"] for b in blocks if b.get("failed_quantities") is not None]
    return {"fits": len(unique), "qualified": sum(d.get("qualified") is True for d in diagnostics),
            "unqualified": sum(d.get("qualified") is False for d in diagnostics),
            "unknown_qualification": sum(d.get("qualified") is None for d in diagnostics),
            "continued": sum(n > 0 for n in continuations), "continuation_known": len(continuations),
            "continuation_events": sum(continuations), "expected_blocks": len(blocks),
            "monitored_blocks": len(checked), "monitored": sum(checked) if checked else None,
            "failed_blocks_known": len(failed), "failed_quantities": sum(failed) if failed else None,
            "max_rhat": extreme("max_rhat", max), "min_bulk_ess": extreme("min_bulk_ess", min),
            "min_tail_ess": extreme("min_tail_ess", min),
            "extrema_blocks": min(len(finite_values(k)) for k in ("max_rhat", "min_bulk_ess", "min_tail_ess")),
            "chains": sorted(set(int(x) for x in finite_values("chains"))),
            "diagnostic_draws": sorted(set(int(x) for x in finite_values("diagnostic_draws")))}


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
        self.horizons = load(self.source / "horizon_curves.json")
        self.horizon_index = {}
        for row in self.horizons:
            key = (row["track"], row["candidate_id"], row["seed"])
            if key in self.horizon_index:
                raise ValueError("Duplicate cohort in compact horizon curves")
            self.horizon_index[key] = row
        self.source_manifest = source_manifest = load(self.source / "source_manifest.json")
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
               "and scaling use the final two thirds of the conditioning prefix, with scale floor 0.04. The d-1 "
               "nonintercept temporal features are then divided by sqrt(d-1). The spectral "
               "radius is measured separately and must not be confused with the configured operator norm.", "small")
        self.p("The p-1 nonintercept spatial dictionary columns are centered on permitted training coordinates, "
               "divided by their training-geometry standard deviations floored at 10⁻⁶, and divided by sqrt(p-1); "
               "the intercept remains one. Both the radial-basis width and residual range ell use raw unit-circle "
               "chord-distance units, whose maximum is two, rather than physical arc-length units. These "
               "dimension normalizations are part of the accepted coefficient-prior interpretation.", "small")
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
            ["Precision and freeze", "Four repeated 512-path ensembles on two fixed cases for each eligible promoted main candidate. Candidate-specific near-tie tolerance; lower total readout coefficient count p × d breaks near ties."],
            ["Confirmation", "Freeze first; evaluate on the other 14 trajectories, all20 validation origins. No reselection on these outcomes."],
            ["Model-seed sensitivity", "Additional seeds 20261106, 20261107 on trajectories 0, 5, 10, 15 and eight origins; matched main-seed references."],
        ], [1.25, 3.75])
        self.p("Selection minimizes episode-balanced CRPS, with equal weights for the mean losses in leads 1-10, 11-25, "
               "26-50 and 51-100. Reported ordinary 100-step CRPS and RMSE are distinct summaries. MCMC keeps four chains, "
               "1000 warmup and 2000 retained states per chain, with one bounded 2000-draw continuation. Qualification "
               "requires rank-normalized split R-hat strictly below 1.01 and bulk/tail ESS ≥ 400 for monitored quantities [3].")
        self.p("All sampled hyperstates are checked. All readout coefficients and ten scale/readout/predictive functionals "
               "are checked using 512 evenly spaced retained states per chain. Predictive precision repetitions measure "
               "conditional simulation variability; they are not substitutes for chain convergence or trajectory-level uncertainty.")

    def metric_definitions(self):
        self.heading("3. Reading the scores and reliability tables")
        self.p("All field values, errors, interval widths and unnormalized scores use the physical units of the clean "
               "simulated field. Native point forecasts are means of complete posterior predictive paths. Point references "
               "have no predictive distribution: their CRPS, coverage, width and joint distribution scores are explicitly undefined.")
        self.table(["Quantity", "Exact interpretation in this experiment"], [
            ["RMSE and nRMSE", "Pool squared errors over origins, horizons and target sites within each trajectory, then take its square root. Average these trajectory RMSEs equally. For nRMSE, first divide each trajectory's RMSE by its permitted training standard deviation. This is not an aggregate RMSE divided by one pooled scale."],
            ["MAE and CRPS", "Average site/horizon/origin losses within each trajectory, then trajectories equally. CRPS is the empirical V-statistic: mean |X-y| minus one half of the mean over all ordered path pairs |X-X'|, including diagonal pairs. No iid finite-ensemble bias correction is claimed [4]."],
            ["50%, 80%, 90%, 95% intervals", "Central empirical predictive quantiles. Coverage is the fraction of clean targets between the inclusive endpoints; width is the mean upper-minus-lower endpoint. These are marginal site/lead intervals, not simultaneous field or path bands."],
            ["Spatial energy", "At each lead, [mean ||X-y||₂ minus one half mean ||X-X'||₂] divided by sqrt(number of target sites). The second term uses 1024 independently sampled path-index pairs, with replacement and diagonal pairs allowed, seed 20261105. It approximates the empirical energy score and can be slightly negative from Monte Carlo error [4]."],
            ["Spatial variogram", "Weighted sum over fixed geometry-only site pairs of (|y_i-y_j|^0.5 minus mean |X_i-X_j|^0.5)². Up to 128 unordered pairs use normalized inverse unit-circle chord-distance weights; pair seed 20261105. Smaller is better. The score assesses spatial increments and is insensitive to a common additive field shift [8]."],
        ], [1.1, 4.1])
        self.p("Prediction time and censoring", "h2")
        self.p("At each origin, valid prediction time (VPT) is the first lead h whose spatial RMSE, divided by that "
               "trajectory's permitted training standard deviation, exceeds 0.4. The reported physical time is 0.25h. "
               "If no crossing occurs by lead 100, the stored time is 25 and the outcome is right-censored. A crossing "
               "at lead 100 also has time 25 but is not censored. Tables report the <b>mean horizon-restricted VPT</b> "
               "and the censored-origin fraction, averaged over the same complete trajectories. This is not an uncensored "
               "mean first-crossing time, a survival-model estimate or a Lyapunov-time normalization.")
        self.p("Reliability and interval populations", "h2")
        self.p("Finite divergence means that at least one lead's spatial RMSE exceeds 10 training standard deviations. "
               "It is distinct from a nonfinite/numerical rollout failure; finite extreme losses remain in the scores. "
               "Accuracy requires all scheduled origins of a trajectory to be available and, for native distributions, "
               "inference-qualified. The full-schedule census separately retains failed, unqualified and pending outcomes.")
        self.p("Square-bracket intervals around aggregate metrics are 95% trajectory-cluster bootstrap intervals, "
               "conditional on the selected configurations and complete trajectories. They are not the 50–95% predictive "
               "interval levels. An observed zero-event bootstrap interval does not establish zero event probability "
               "on new trajectories. Metric definitions are implemented in applications/ks_metrics.py and scoring.py.", "small")

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
            self.extended_track_metrics(track, rows)

    def extended_track_metrics(self, track, rows):
        self.heading(f"3{track} continued. Scale, spatial dependence and prediction time")
        self.p("Means and 95% trajectory-bootstrap intervals use each method's complete-trajectory population. "
               "The complete count and the preceding full-schedule census must accompany every accuracy comparison. "
               "No missing score is replaced with zero.", "small")
        self.table(["Method / role", "Complete", "nRMSE", "Spatial energy", "Spatial variogram"], [
            [role(r), f"{r['complete_trajectories']}/{r['scheduled_trajectories']}",
             interval(r["metrics"]["nrmse"]),
             "Point only" if "cheap_anchor" in r.get("roles", []) else interval(r["metrics"]["spatial_energy"]),
             "Point only" if "cheap_anchor" in r.get("roles", []) else interval(r["metrics"]["spatial_variogram"])]
            for r in rows], [1.9, .65, 1.5, 1.5, 1.5])
        self.table(["Method / role", "Complete", "Mean restricted VPT (physical time)", "Censored origins (%)"], [
            [role(r), f"{r['complete_trajectories']}/{r['scheduled_trajectories']}",
             interval(r["metrics"]["vpt_physical"], 3),
             percent_interval(r["metrics"]["vpt_censoring_fraction"])] for r in rows], [1.9, .65, 1.75, 1.45])
        self.p("Restricted VPT is capped at 25 physical time units. Its censoring fraction is the proportion of "
               "included origins without a threshold crossing by lead 100. A large censored fraction requires "
               "particular care: the uncensored mean beyond this horizon is not identified here.", "small")
        self.heading(f"3{track} continued. Marginal interval calibration")
        native = [r for r in rows if "cheap_anchor" not in r.get("roles", [])]
        if native:
            self.table(["Method / role", "Complete", "Nominal level", "Coverage (%)", "Mean width"], [
                [role(r), f"{r['complete_trajectories']}/{r['scheduled_trajectories']}", f"{level}%",
                 percent_interval(r["metrics"][f"coverage{level}"]),
                 interval(r["metrics"][f"width{level}"])] for r in native for level in (50, 80, 90, 95)],
                [1.9, .65, .75, 1.55, 1.55])
        else:
            self.p("No native predictive-distribution cohort is present in this snapshot.")
        self.p("Coverage refers to clean field values and width uses physical field units. Brackets are 95% "
               "trajectory-bootstrap intervals for the aggregate, while the nominal column identifies the distinct "
               "predictive interval level. Point-only anchors have no interval coverage or width; none is manufactured.", "small")

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
                (f"search_{track}.png", f"6{track}. Search evidence", "Configuration indices map to plot_index and the exact candidate_id within each track/stage in search_candidates.csv. All evaluated configurations retain their eligibility and failure counts there. Legacy prior references, fixed-prior candidates and prior sensitivity remain separate. These scores are selection data."),
                (f"compute_{track}.png", f"7{track}. Accuracy and compute", "Fitting cost is measured wall time per fitted model. Forecast and search time are reported separately; neither candidate count nor epoch count alone measures comparable opportunity."),
                (f"example_{track}_t01_a6000.png", f"8{track}. Predeclared space-time example", "Trajectory 1, first target 6000 was fixed before confirmation. Only official validation target sites are scored. The site map distinguishes permitted inputs from held-out targets; categorical target rows do not imply interpolation of a fully observed field."),
                (f"intervals_{track}_t01_a6000.png", f"9{track}. Predictive intervals at fixed coordinates", "Coordinates follow the first/middle/last official target-site rule. These are marginal 90% clean-field predictive intervals. They do not describe simultaneous field or path coverage."),
            ]:
                if name in self.r["figures"]:
                    self.heading(title)
                    self.figure(name, 605, caption)

    def selected_lead_values(self, cohort, metric, *, percent=False):
        key = (cohort["track"], cohort["candidate_id"], cohort["seed"])
        row = self.horizon_index.get(key)
        if row is None:
            if cohort["complete_trajectories"]:
                raise ValueError("A completed cohort has no bound horizon summary")
            return ["Undefined"] * len(LEADS)
        if row["complete_trajectory_ids"] != cohort["complete_trajectory_ids"]:
            raise ValueError("Horizon and aggregate score populations disagree")
        curve = row["curves"].get(metric)
        if curve is None:
            raise ValueError("Requested metric is absent from a bound horizon summary")
        mean = curve.get("mean")
        expected_n = cohort["metrics"][metric]["n"]
        if curve["n"] != expected_n:
            raise ValueError("Horizon and aggregate metric denominators disagree")
        if mean is None:
            if expected_n:
                raise ValueError("A populated horizon summary has no values")
            return ["Undefined"] * len(LEADS)
        if len(mean) != 100:
            raise ValueError("The selected-lead appendix requires the frozen 100-lead curve")
        return [fmt(mean[h-1] * (100 if percent else 1), 1 if percent else 4) for h in LEADS]

    def horizon_appendix(self):
        for track in self.r["tracks"]:
            rows = [r for r in self.r["confirmation"] if r["track"] == track
                    and any(x in r["roles"] for x in ("main", "anchor", "cheap_anchor"))]
            self.heading(f"Appendix A. Track {track}: predeclared forecast leads")
            if not rows:
                self.p("Confirmation horizon summaries are pending. No values are substituted.")
                continue
            headings = ["Method / metric"] + [f"h={h}\ntime={.25*h:g}" for h in LEADS]
            point_rows = []
            for r in rows:
                for metric in ("rmse", "mae"):
                    point_rows.append([f"{role(r)}\n{metric.upper()} | n={r['complete_trajectories']}/"
                                       f"{r['scheduled_trajectories']}"] + self.selected_lead_values(r, metric))
            self.table(headings, point_rows, [2.25] + [1] * len(LEADS))
            self.p("Saved trajectory-first horizon means are shown; time = 0.25 × lead in physical units. "
                   "At a fixed lead, RMSE first pools that trajectory's origin/site squared errors and takes their square "
                   "root, then averages trajectories. Pointwise 95% trajectory-bootstrap intervals are retained in "
                   "horizon_curves.json and the horizon figures. The rows retain each method's complete count; this "
                   "table does not replace the paired comparisons or full-schedule census.", "small")
            native = [r for r in rows if "cheap_anchor" not in r["roles"]]
            if native:
                self.heading(f"Appendix A. Track {track}: native distribution scores by lead")
                labels = (("crps", "CRPS", False), ("coverage90", "90% coverage (%)", True),
                          ("width90", "90% width", False))
                self.table(headings, [[f"{role(r)}\n{label} | n={r['complete_trajectories']}/"
                                       f"{r['scheduled_trajectories']}"]
                                     + self.selected_lead_values(r, metric, percent=percent)
                                     for r in native for metric, label, percent in labels], [2.25] + [1] * len(LEADS))
                self.p("These are saved marginal CRPS, coverage and interval-width means at the same six leads. "
                       "Point-only references have no corresponding predictive-distribution values. All four interval "
                       "levels appear in the per-track aggregate calibration tables; all 100-lead curves remain available "
                       "in the bound compact JSON. Prior sensitivity remains a separate analysis.", "small")

    def inference_appendix(self):
        self.heading("Appendix B. Actual inference qualification and effort")
        self.p("The following summaries read only fit_diagnostics.json. A reused fitted artifact is counted once by "
               "fit_key. The all-fit population includes screening, promotion, confirmation and seed sensitivity "
               "artifacts available in this snapshot; it is not a denominator of independent scientific replicates. "
               "Qualification is a numerical diagnostic and does not establish predictive accuracy.")
        groups = [(track, [r for r in self.fits if r["track"] == track]) for track in self.r["tracks"]]
        self.table(["All available fits", "Unique fits", "Qualified", "Unqualified", "Unknown", "Continued / known"], [
            [track, (d := diagnostic_summary(rows))["fits"], d["qualified"], d["unqualified"],
             d["unknown_qualification"], f"{d['continued']}/{d['continuation_known']}"] for track, rows in groups],
            [1.25, .8, .9, .95, .75, 1.25])
        rows = []
        for track, fits in groups:
            for block in DIAGNOSTIC_BLOCKS:
                d = diagnostic_summary(fits, block)
                rows.append([f"{track}: {block}", f"{d['monitored_blocks']}/{d['expected_blocks']}",
                             "Unknown" if d["monitored"] is None else f"{d['monitored']:,}",
                             fmt(d["max_rhat"]), fmt(d["min_bulk_ess"], 1), fmt(d["min_tail_ess"], 1),
                             "Unknown" if d["failed_quantities"] is None else d["failed_quantities"]])
        self.table(["Track / block", "Receipts / fits", "Checks", "Max R-hat", "Min bulk ESS", "Min tail ESS", "Failed checks"],
                   rows, [1.7, .85, 1, 1, 1, 1, .85])
        self.p("Checks sum the numbers of monitored scalar quantities across distinct fitted artifacts. Failed checks "
               "are recorded per quantity and block, not failed forecasts. Extrema include unqualified fits; they are "
               "not asserted to meet the gate. Missing compact diagnostics remain unknown. These are post-continuation "
               "diagnostics when a continuation was used.", "small")
        observed = []
        for block in DIAGNOSTIC_BLOCKS:
            d = diagnostic_summary(self.fits, block)
            observed.append([block, ", ".join(map(str, d["chains"])) or "Unknown",
                             ", ".join(map(str, d["diagnostic_draws"])) or "Unknown",
                             f"{d['extrema_blocks']}/{d['expected_blocks']}"])
        self.table(["Block", "Recorded chain counts", "Diagnostic states per chain", "Finite extrema / block receipts"],
                   observed, [1.3, 1.1, 1.7, 1.7])
        self.p("The gate requires four chains, rank-normalized split R-hat strictly below 1.01 and bulk/tail ESS ≥ 400. "
               "Hyperparameters use all retained sampled states. Every readout coefficient and the ten monitored "
               "functionals are checked on 512 evenly spaced retained states per chain; this disclosed diagnostic "
               "subsample does not exclude coefficient directions. The one allowed continuation adds 2000 retained "
               "states per chain to the original 2000. Warmup, chains, diagnostic states, predictive paths and model "
               "seeds are separate quantities; no HMC-specific diagnostics apply to this sampler.", "small")
        selected = [r for r in self.r["confirmation"] if "cheap_anchor" not in r["roles"]]
        self.heading("Appendix B continued. Selected confirmation fits")
        if not selected:
            self.p("Selected confirmation fits are pending in this snapshot; the preceding screening-inclusive "
                   "counts must not be read as final selected-model qualification.")
            return
        summaries = []
        for r in selected:
            fits = [f for f in self.fits if (f["track"], f["candidate_id"], f["seed"]) ==
                    (r["track"], r["candidate_id"], r["seed"])
                    and f["trajectory"] in self.r["confirmation_roster"]]
            d = diagnostic_summary(fits)
            if d["fits"] != r["fits_available"] or d["qualified"] != r["qualified_fits"]:
                raise ValueError("Selected confirmation compact fit counts disagree with the score census")
            summaries.append((r, d))
        self.table(["Track / role", "Fits /14", "Qualified", "Unqualified / unknown", "Continued / known"], [
            [f"{r['track']}: {role(r)}", f"{d['fits']}/{r['scheduled_trajectories']}", d["qualified"],
             f"{d['unqualified']} / {d['unknown_qualification']}", f"{d['continued']}/{d['continuation_known']}"]
            for r, d in summaries], [2.4, .8, .8, 1.2, 1.2])
        self.table(["Track / role", "Checks", "Max R-hat", "Min bulk ESS", "Min tail ESS", "Failed checks"], [
            [f"{r['track']}: {role(r)}", "Unknown" if d["monitored"] is None else f"{d['monitored']:,}",
             fmt(d["max_rhat"]), fmt(d["min_bulk_ess"], 1), fmt(d["min_tail_ess"], 1),
             "Unknown" if d["failed_quantities"] is None else d["failed_quantities"]]
            for r, d in summaries], [2.4, 1.1, 1, 1, 1, .9])
        self.p("These rows restrict the compact fits to the frozen confirmation roster and each cohort's seed/configuration. "
               "They include available unqualified fits, even when no predictive score is eligible. Missing or failed "
               "fits are not added to the diagnostic extrema; their blocked-origin counts remain in the main reliability "
               "census. Fixed point references require no MCMC qualification.", "small")

    def fixed_reference_appendix(self):
        anchors = self.source_manifest.get("anchors")
        if not anchors:
            return
        self.heading("Appendix C. Exact fixed point-reference configurations")
        self.p("These references retain the original experiment's selected settings and receive no new tuning. "
               "The contiguous track transfers the declared A configuration and refits permitted C training data. "
               "All origins use the common 512-state history. Vector recurrences advance their native observed/latent "
               "state; the periodic decoder is applied to outputs, without hidden-truth feedback or repeated smoothing projection.")
        plan = anchors["plan"]
        rows = []
        for track in plan["tracks"]:
            for model in plan["models"]:
                c = plan["configurations"][track][model]
                rows.append([f"{track}: {MODEL_NAMES.get(model, model)}", c["id"],
                             c["original_source_track"], str(plan["model_seeds"][model]),
                             json.dumps(c["config"], sort_keys=True, separators=(", ", ": "))])
        self.table(["Track / reference", "Original ID", "Original track", "Seed", "Exact effective configuration"],
                   rows, [1.55, .9, .6, .85, 3.1])
        self.p("A deterministic family's seed is an unused identity field, not a replicated fit. Exact configuration "
               "objects and the fixed-reference plan identity are retained in source_manifest.json. No Gaussian "
               "distribution is added to these point forecasts.", "small")

    def operations(self):
        self.heading("10. Computation, acceptance and limitations")
        ops = self.r["operations"]
        budget = self.r["budget"]
        total_hours = budget["total_worker_seconds_cap"] / 3600
        search_hours = budget["search_worker_seconds_cap"] / 3600
        elapsed_hours = budget["wall_seconds_cap"] / 3600
        reserved_hours = total_hours - search_hours
        self.table(["Stage", "Worker wall hours"], [[k, fmt(v/3600,3)] for k,v in ops["attempt_worker_seconds_by_stage"].items()] +
                   [["Recorded preparation", fmt(ops["preparation_worker_seconds"]/3600,3)],
                    ["Cumulative attempt + preparation", fmt((ops["attempt_worker_seconds"]+ops["preparation_worker_seconds"])/3600,3)]], [3,2])
        references = [r for r in self.r["confirmation"] if "cheap_anchor" in r.get("roles", [])]
        if references:
            self.table(["Separate auxiliary work", "Measured seconds", "Timing scope"], [
                ["Fixed point-reference fitting", fmt(sum(r["fit_seconds"] for r in references), 2),
                 "Sum of available fitted-model receipts; excluded from main worker ledger"],
                ["Fixed point-reference forecasting", fmt(sum(r["forecast_seconds"] for r in references), 2),
                 "Sum of recorded origin forecasts, including recorded failures; excluded from main worker ledger"]],
                [1.6, 1, 2.8])
        self.p("The fixed point-reference evaluation and selected-artifact acceptance checks are separate auxiliary "
               "work, not part of the main campaign's recorded worker-attempt total. The reference tool's default "
               "900-second invocation limit is checked before each whole trajectory/model job; an admitted job can "
               "finish beyond that threshold. Reference fit/forecast timings exclude orchestration overhead and are "
               "not substituted for an end-to-end invocation time. The accepted reference configurations are fixed "
               "before these auxiliary evaluations; no new tuning is performed.", "small")
        self.p("The environment has an 8-core CPU quota and an 8 GiB container memory limit; no GPU was used. "
               f"Up to {budget['workers']} model workers use {budget['threads_per_worker']} numerical thread(s) each. "
               "The measured 512-unit pilot required 64.82 s for a qualified fit, "
               "5.53 s for one 256-path forecast and 537,538,560 bytes peak RSS. Derived matrices use the accepted separable "
               "structure; the full space-time tensor design and a giant readout covariance are not materialized.")
        self.p(f"The active configured admission thresholds are {total_hours:g} worker hours overall, "
               f"{search_hours:g} worker hours for search, and {elapsed_hours:g} elapsed hours. The difference "
               f"({reserved_hours:g} worker hours) is reserved for confirmation and diagnostics. The campaign clock "
               f"starts at {html.escape(budget['campaign_started_utc'])}; resuming does not restart it. "
               f"Already admitted jobs may finish after a threshold. At {budget['workers']} workers with "
               f"{budget['job_timeout_seconds']:g}-second per-job timeouts, the maximum additional admitted group "
               f"represents {budget['workers'] * budget['job_timeout_seconds']:g} worker seconds, plus shutdown overhead. "
               "Actual recorded use, not the nominal threshold alone, determines budget compliance.")
        extension = budget.get("operational_extension")
        if extension:
            if (extension.get("one_prospective_extension") is not True
                    or extension.get("frozen_science_unchanged") is not True
                    or extension.get("confirmation_scores_used_for_decision") is not False
                    or extension["original_campaign_started_utc"] != budget["campaign_started_utc"]
                    or extension["search_worker_seconds_cap_unchanged"] != budget["search_worker_seconds_cap"]):
                raise ValueError("Operational budget extension lacks the recorded unchanged-science/time boundary")
            self.p("One prospective operational extension", "h2")
            self.table(["Scheduling policy", "Total worker hours", "Search worker hours", "Elapsed hours"], [
                ["Original executed policy", extension["original_total_worker_seconds_cap"] / 3600,
                 extension["search_worker_seconds_cap_unchanged"] / 3600,
                 extension["original_wall_seconds_cap"] / 3600],
                ["Active extended policy", total_hours, search_hours, elapsed_hours]], [1.8, 1.2, 1.2, 1.1])
            self.p("The extension was authorized prospectively from receipt timing and remaining-work estimates. "
                   "Confirmation performance scores were not used for that decision. It preserves the original "
                   "campaign start and all spent worker time; it does not restart the clock, reopen selection, "
                   "change the scientific configuration or erase failures. Migration rebinds the experiment and "
                   "freeze identities to the operational configuration while retaining the frozen choices and "
                   "original freeze timestamp. The source run and previous configuration remain evidence.")
            self.p("Recorded reason: " + html.escape(budget["extension_reason"]), "small")
            self.table(["Extension provenance", "Recorded value"], [
                ["Authorization recorded (UTC)", extension["authorized_utc"]],
                ["Timing-only receipt SHA-256", extension["timing_receipt_sha256"]],
                ["Original frozen-selection identity", extension["frozen_selection_identity"]],
                ["Previous configuration SHA-256", extension["previous_configuration_sha256"]]], [1.5, 3.5])
            if extension.get("authorization_timestamp_scope"):
                self.p(html.escape(extension["authorization_timestamp_scope"]), "small")
        self.p("Scheduling also requires the 1.5× worker-memory estimate to fit the combined allowance and 65% of "
               f"container-aware headroom. The disk reserve is {budget['disk_reserve_bytes']:,} bytes. Cache advice releases pages of closed "
               "fit files after proved child termination; it neither deletes checkpoints nor relaxes admission.")
        self.p("An early mutable-ledger inconsistency was repaired by verifying 208 immutable preflight artifacts, "
               "carrying their identities and prior time forward, and adopting immutable terminal receipts as authority. "
               "The responsible external mechanism was not established. No scientific candidate, prior, data, forecast "
               "or score was changed by that recovery. The original run remains evidence.")
        a = self.r.get("selected_invariance")
        if a:
            self.p(f"<b>Selected-artifact acceptance:</b> {a['passed_cases']}/{a['expected_cases']} cases passed; "
                   f"failed={a['failed_cases']}, complete={a['complete']}. Tests fix fitted parameters and randomness, "
                   "perturb hidden histories and future labels, and separately refit preprocessing/features on unchanged "
                   "permitted training inputs. They establish output separation, not future-free physical disk access.")
            if a.get("last_invocation_seconds") is not None:
                self.p(f"The recorded latest acceptance-check invocation took {fmt(a['last_invocation_seconds'], 2)} "
                       "seconds, separately from the campaign worker ledger. This field is the latest invocation, "
                       "not an assertion that earlier attempts required zero time.", "small")
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
            ("8", "Scheuerer and Hamill (2015). Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities.", "https://repository.library.noaa.gov/view/noaa/22327/"),
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
        self.introduction(); self.contract(); self.search(); self.metric_definitions(); self.result_pages()
        self.parameter_pages(); self.sensitivity_pages(); self.figure_pages(); self.operations()
        self.horizon_appendix(); self.inference_appendix(); self.fixed_reference_appendix(); self.references()
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
