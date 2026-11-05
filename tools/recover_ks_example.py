#!/usr/bin/env python3
"""Recompute one predeclared KS example from an identity-verified retained fit.

This is a reconstruction, not recovery of original prediction-file bytes.
No model fitting, hyperparameter selection, or new aggregate ranking occurs.

Run from the repository:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONPATH=src /path/to/env/bin/python \
  tools/recover_ks_example.py --workspace /path/to/recovery_workspace

--render-only verifies and reuses this script's completed numerical receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import tempfile
from datetime import datetime, timezone

# This standalone reconstruction has one numerical worker and one encoder.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from bayes_stesn.io import atomic_npz, content_hash, seed_for
from bayes_stesn.applications.ks_runtime import atomic_json
from bayes_stesn.applications.ks_data import KSAdapter, IO_LIMITATION
from bayes_stesn.applications.ks_native import NativeFit, _computation_identity
from bayes_stesn.applications.ks_metrics import score
from bayes_stesn.benchmark.classical import _panel_state

FIT_SHA = "cfe7b6598deaf76beb7198232ac8a98c17eb22861956cf811808b78ea0d38049"
FREEZE_SHA = "90e9a5e82e077eae15c19ae887496f1ec4b4fc706925d961ae539852efd699e8"
FREEZE_ID = "ca6d8dac49208767f0073a747a3fbc7ee37648715a22abc32e539378b75756dc"
EXPERIMENT_ID = "7604e42da7265f3d3891aa73491affc2d851738003bf8264647ac4a01ba182e5"
NATIVE_ID = "31ed706dc9abc19821f7d11b60afecd9aa43096ce5ee064e1b7b42627c7d92c1"
RUNNER_FIT_ID = "5d53d036c33a862f26600a48956abb1282d0e5368282290d4aa7ab13ce76b280"
MODEL, MODEL_SEED, FORECAST_SEED = "bayes_stesn", 20261105, 11622026
WORKFLOW_DISK_RESERVE = 12789245542  # Fixed from the recovery workspace's initial free space.
A, H, HISTORY, PATHS, BATCH = 8000, 100, 128, 512, 64
LEVELS = np.array([.025, .05, .1, .25, .5, .75, .9, .95, .975])
CASE = {"phase": "core", "track": "A", "trajectory": 0, "first_target": A,
        "model": MODEL, "model_seed": MODEL_SEED, "history": HISTORY,
        "horizon": H, "paths": PATHS, "path_batch": BATCH}
HISTORICAL = {
    "source": "selected numerical values retained in the pre-outage conversation; not original prediction bytes",
    "leads": [1, 5, 10, 25, 50, 100],
    "rmse": [.15484099253493358, .9509984346558278, 1.0715633812931733,
             1.4239297730157205, 1.381765052606371, 1.2289178699290966],
    "width90": [.13936063906352647, .8948377497825238, 2.49574780221798,
                5.139202278780546, 5.937455784413632, 5.911421601461554],
    "mean_abs_max": 3.737863335227227,
    "truth_abs_max": 3.0684838277703594,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resources(path):
    cg = Path("/sys/fs/cgroup")
    def number(name):
        p = cg/name
        value = p.read_text().strip() if p.exists() else None
        return int(value) if value and value.isdigit() else value
    limit, current = number("memory.max"), number("memory.current")
    return {"cpu_affinity_count": len(os.sched_getaffinity(0)),
            "cpu_quota": number("cpu.max"),
            "container_memory_limit_bytes": limit,
            "container_memory_current_bytes": current,
            "container_memory_available_bytes": limit-current if isinstance(limit, int) and isinstance(current, int) else None,
            "disk_free_bytes": shutil.disk_usage(path).free,
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            "largest_child_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*1024}


def verify(workspace):
    fit_path = workspace/"evidence/ks_core_A_bayes_stesn_t00_s20261105.npz"
    freeze_path = workspace/"evidence/ks_frozen_specification_full.json"
    checks = {}
    def require(name, okay):
        checks[name] = bool(okay)
        if not okay:
            raise ValueError("Reconstruction binding failed: " + name)
    require("fit_file_sha256", sha(fit_path) == FIT_SHA)
    require("freeze_file_sha256", sha(freeze_path) == FREEZE_SHA)
    frozen = json.loads(freeze_path.read_text())
    require("freeze_identity", frozen["identity"] == FREEZE_ID and
            content_hash({k:v for k,v in frozen.items() if k != "identity"}) == FREEZE_ID)
    require("experiment_identity", frozen["experiment_identity"] == EXPERIMENT_ID)
    repo = Path(__file__).resolve().parents[1]
    package = repo/"src/bayes_stesn"
    source_files = {name:sha(package/name) for name in frozen["source"] if name != "runtime"}
    require("all_51_frozen_numerical_source_files", len(source_files) == 51 and
            source_files == {k:v for k,v in frozen["source"].items() if k != "runtime"})
    used_packages = {name:importlib.metadata.version(name)
                     for name in ("numpy","scipy","h5py","threadpoolctl")}
    require("bayesian_and_reader_package_versions", all(used_packages[name] ==
            frozen["source"]["runtime"]["packages"][name] for name in used_packages))
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    require("frozen_case", {"track":"A", "trajectory":0, "first_target":A} in frozen["plot_cases"])
    require("frozen_schedule", (frozen["history"], frozen["horizon"], frozen["paths"], frozen["model_seed"])
            == (HISTORY, H, PATHS, MODEL_SEED) and A in frozen["test_origins"])
    config = frozen["selected"]["A"][MODEL]["config"]
    experiment = json.loads((repo/"configs/applications/ks_experiment.json").read_text())
    require("forecast_seed_and_batch", experiment["forecast_seed"] == FORECAST_SEED and experiment["path_batch"] == BATCH)
    fit = NativeFit.load(fit_path)
    require("native_identity", fit.identity == NATIVE_ID)
    require("native_computation", _computation_identity(MODEL) == fit.computation)
    require("qualified_original_fit", fit.diagnostics.get("qualified") is True)
    require("same_model_seed", fit.model_id == MODEL and fit.seed == MODEL_SEED)
    require("selected_effective_config", json.loads(json.dumps(fit.config)) == config)
    adapter = KSAdapter(workspace/"cache", "interleaved", "core")
    raw_path = adapter.folder/adapter.reader.files[0]["path"]
    require("raw_trajectory_sha256", sha(raw_path) == adapter.reader.files[0]["sha256"])
    train = adapter.training(0)
    input_identity = content_hash({"panel":_panel_state(train.panel), "metadata":dict(train.panel.metadata)})
    require("permitted_training_identity", input_identity == fit.input_identity)
    require("stored_training_values", np.array_equal(train.panel.y, fit.training.y))
    require("stored_training_geometry", np.array_equal(train.panel.coordinates, fit.training.coordinates))
    require("stored_training_times", np.array_equal(train.panel.times, fit.training.times))
    require("training_bounds_and_sites", train.panel.y.shape == (6000,39) and
            train.transform.fit_time_bounds == (0,6000) and train.transform.training_value_count == 234000)
    runner_id = content_hash({"model":MODEL, "track":"A", "phase":"core",
        "training":dict(train.identity), "config":config, "model_seed":MODEL_SEED, "source":frozen["source"]})
    require("runner_fit_identity", runner_id == RUNNER_FIT_ID)
    window = adapter.model_window(0,A,"test",history=HISTORY,horizon=H)
    require("history_and_target_shapes", window.panel.y.shape == (HISTORY,39) and len(window.target_site_ids) == 12)
    require("permitted_history_only", not np.intersect1d(window.site_ids,window.target_site_ids).size)
    require("history_target_time_indices", np.array_equal(window.panel.times,np.arange(A-HISTORY,A)*.25) and
            np.array_equal(window.target_time,np.arange(A,A+H)*.25))
    stream = seed_for(FORECAST_SEED,"core","A",0,MODEL,MODEL_SEED,A)
    binding = {"freeze_identity":FREEZE_ID, "freeze_file_sha256":FREEZE_SHA,
        "experiment_identity":EXPERIMENT_ID, "fit_sha256":FIT_SHA,
        "native_fit_identity":NATIVE_ID, "runner_fit_identity":runner_id,
        "native_input_identity":input_identity, "training_identity":dict(train.identity),
        "original_full_study_numerical_identity_sha256":content_hash(frozen["source"]),
        "verified_source_file_hashes":source_files, "verified_runtime_packages":used_packages,
        "native_computation":fit.computation,
        "recovery_scope":"family-scoped standalone Bayesian reconstruction; not an original full-runner job resume",
        "full_study_runtime_reproduced":False,"torch_version_in_recovery_environment":torch_version,
        "original_full_runner_guard_error":"PackageNotFoundError: No package metadata was found for torch",
        "scope_reason":"Exact native Bayesian runtime, permitted data, configuration and all 51 frozen source files verified; no neural module is executed. Original runner guards are unchanged.",
        "configuration":config, "configuration_id":frozen["selected"]["A"][MODEL]["id"],
        "case":CASE, "forecast_seed_root":FORECAST_SEED, "forecast_stream":stream,
        "checks":checks, "check_count":len(checks)}
    return adapter, train, window, fit, binding


def render_intervals(arrays, output):
    """Use saved quantiles and the original predeclared coordinate rule only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from bayes_stesn.applications.ks_report import INTERVAL_INDICES
    chosen = INTERVAL_INDICES["A"]
    levels = np.asarray(arrays["quantile_levels"])
    def quantile(level):
        index = np.flatnonzero(np.isclose(levels,level,rtol=0,atol=1e-12))
        if len(index) != 1:
            raise ValueError("Missing saved marginal quantile " + str(level))
        return arrays["quantiles"][index[0]]
    lo90,hi90,lo50,hi50 = map(quantile,(.05,.95,.25,.75))
    leads = np.arange(1,H+1)*.25
    fig,axes = plt.subplots(3,1,figsize=(8.6,7.5),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,bottom=.14,top=.85,hspace=.42)
    for ax,index in zip(axes,chosen):
        ax.fill_between(leads,lo90[:,index],hi90[:,index],color="#2166ac",alpha=.14)
        ax.fill_between(leads,lo50[:,index],hi50[:,index],color="#2166ac",alpha=.30)
        ax.plot(leads,arrays["mean"][:,index],color="#2166ac",linewidth=1.5)
        ax.plot(leads,arrays["truth"][:,index],color="black",linewidth=1.2)
        ax.set_title(f"Fixed target position {index}: site {arrays['site_ids'][index]}, x={arrays['x'][index]:.5g}",loc="left",fontsize=10)
        ax.set_ylabel("KS field\n(physical units)");ax.grid(alpha=.18);ax.set_xlim(.25,25)
    axes[-1].set_xlabel("Elapsed physical time (0.25 × forecast lead)")
    handles=[Line2D([],[],color="black",label="Clean truth"),
             Line2D([],[],color="#2166ac",label="Mean of 512 complete paths"),
             Patch(color="#2166ac",alpha=.30,label="50% central predictive interval"),
             Patch(color="#2166ac",alpha=.14,label="90% central predictive interval")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
    fig.suptitle("Strict unseen-site Bayesian ST-ESN: fixed-coordinate predictive intervals\n"
                 "Trajectory 0, first target 8000; recomputed from retained fit",fontsize=12,y=.97)
    fig.text(.5,.895,"Original predeclared target positions 0, 5, 11; clean-field target; saved quantiles only.",ha="center",fontsize=9)
    for fmt in ("png","pdf"):
        buffer=io.BytesIO();fig.savefig(buffer,format=fmt,dpi=180);payload=buffer.getvalue()
        if not payload or (fmt=="pdf" and not payload.startswith(b"%PDF-")):
            raise RuntimeError("Empty or invalid interval figure")
        if fmt=="pdf":subprocess.run(["pdfinfo","-"],input=payload,check=True,capture_output=True)
        final=output/f"recomputed_A_bayes_t00_a8000_intervals.{fmt}"
        with tempfile.NamedTemporaryFile(dir=output,suffix="."+fmt,delete=False) as stream:
            staging=Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(staging,final)
    plt.close(fig)
    return list(chosen)


def render(arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    plt.rcParams.update({"font.size":10, "axes.titlesize":11, "savefig.dpi":180,
                         "pdf.fonttype":42, "ps.fonttype":42})
    y, m, x, ix = (arrays[k] for k in ("truth","mean","x","input_x"))
    order = np.argsort(x); x, y, m = x[order], y[:,order], m[:,order]
    err = m-y; vmax = max(np.abs(y).max(),np.abs(m).max()); emax = np.abs(err).max()
    fig, axes = plt.subplots(1,3,figsize=(13,6),sharey=True)
    fig.subplots_adjust(left=.075,right=.94,bottom=.32,top=.79,wspace=.38)
    for j,(ax,z,title) in enumerate(zip(axes,(y,m,err),("Clean truth","Predictive mean","Signed error: mean − truth"))):
        limit = vmax if j < 2 else emax
        image = ax.imshow(z.T, origin="lower",aspect="auto",interpolation="nearest",
                          extent=(.125,25.125,-.5,11.5),cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.set_title(title); ax.set_xlabel("Lead time (physical units)")
        ax.set_xticks([.25,5,10,15,20,25]); ax.set_xticklabels(["0.25","5","10","15","20","25"])
        ax.set_yticks(np.arange(12))
        if j == 0:
            ax.set_yticklabels([f"{v:.2f}" for v in x]); ax.set_ylabel("12 withheld sites: physical x")
        else:
            ax.tick_params(labelleft=False)
        fig.colorbar(image,ax=ax,fraction=.047,pad=.035,label="KS field" if j<2 else "Signed error")
    fig.suptitle("Strict unseen-site KS • Bayesian ST-ESN • trajectory 0, first target 8000\n"
                 "Recomputed from retained fit; 512 complete predictive paths",y=.96)
    rug = fig.add_axes([.12,.14,.77,.025])
    rug.scatter(ix,np.zeros_like(ix),marker="|",color=".50",s=90,label="39 permitted input sites")
    rug.scatter(x,np.zeros_like(x),marker="o",facecolors="none",edgecolors="#a12622",s=35,label="12 scored target sites")
    rug.set_xlim(0,22); rug.set_yticks([]); rug.set_xticks([0,5.5,11,16.5,22])
    rug.tick_params(axis="x",labelsize=9); rug.set_xlabel("Physical coordinate x on the periodic domain",labelpad=1)
    for spine in rug.spines.values(): spine.set_visible(False)
    rug.legend(loc="lower center",bbox_to_anchor=(.5,1.08),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.012,"Rows show official target sites only; no error is evaluated at the other locations.",
             ha="center",fontsize=9)
    # Render into memory and atomically publish complete figures. This also
    # refuses silent empty PDF output before another process can consume it.
    for fmt in ("png","pdf"):
        buffer = io.BytesIO()
        fig.savefig(buffer,format=fmt)
        payload = buffer.getvalue()
        if not payload or (fmt == "pdf" and not payload.startswith(b"%PDF-")):
            raise RuntimeError("Matplotlib produced an empty or invalid " + fmt + " figure")
        if fmt == "pdf":
            subprocess.run(["pdfinfo","-"],input=payload,check=True,capture_output=True)
        final = output/f"recomputed_A_bayes_t00_a8000_heatmap.{fmt}"
        with tempfile.NamedTemporaryFile(dir=output,suffix="."+fmt,delete=False) as stream:
            staging = Path(stream.name); stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(staging,final)
    plt.close(fig)
    render_intervals(arrays,output)

    fig, ax = plt.subplots(figsize=(10,5.6),dpi=120)
    fig.subplots_adjust(left=.09,right=.97,top=.76,bottom=.25)
    ax.set_xlim(0,22); ax.set_ylim(-vmax*1.12,vmax*1.12)
    ax.set_ylabel("Clean KS field (physical units)"); ax.set_xlabel("Periodic physical coordinate x",labelpad=18)
    ax.set_xticks([0,5.5,11,16.5,22]); ax.grid(alpha=.15)
    truth, = ax.plot(x,y[0],"o",color="black",markersize=5,label="Truth at 12 withheld sites")
    prediction, = ax.plot(x,m[0],"D",color="#2166ac",markersize=5,label="Predictive mean at those sites")
    ax.scatter(ix,np.full_like(ix,-.15),transform=ax.get_xaxis_transform(),marker="|",
               color=".5",s=80,clip_on=False)
    fig.legend(handles=[truth,prediction],loc="upper center",bbox_to_anchor=(.53,.865),
               ncol=2,frameon=False,fontsize=9)
    heading = fig.suptitle("",y=.98,fontsize=13)
    fig.text(.5,.10,"39 grey rug marks: permitted input sites; markers above: 12 withheld evaluation sites.",
             ha="center",fontsize=9)
    fig.text(.5,.045,"Trajectory 0 • first target 8000 • history 128 • 512 complete paths\n"
             "Recomputed from retained original fit; no refit or reselection.",ha="center",fontsize=9)
    movie = output/"recomputed_A_bayes_t00_a8000.mp4"
    writer = FFMpegWriter(fps=10,codec="libx264",bitrate=1400,
        metadata={"title":"Recomputed predeclared strict-site KS Bayesian example",
                  "comment":"New reconstruction from retained original fit; official 12 target sites only"},
        extra_args=["-pix_fmt","yuv420p","-movflags","+faststart","-preset","veryfast",
                    "-threads","1","-filter_threads","1"])
    with writer.saving(fig,str(movie),dpi=120):
        for k in range(H):
            truth.set_ydata(y[k]); prediction.set_ydata(m[k])
            heading.set_text(f"Strict unseen-site Bayesian ST-ESN | lead {k+1:3d}/100\n"
                f"Elapsed time = {.25*(k+1):.2f}; target time = {arrays['times'][k]:.2f}")
            writer.grab_frame()
    plt.close(fig)
    probe = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=codec_name,width,height,avg_frame_rate,nb_read_frames,pix_fmt,duration",
        "-of","json",str(movie)],check=True,capture_output=True,text=True)
    result = json.loads(probe.stdout)["streams"][0]
    if not (result["codec_name"]=="h264" and result["pix_fmt"]=="yuv420p" and
            result["nb_read_frames"]=="100" and result["avg_frame_rate"]=="10/1" and
            abs(float(result["duration"])-10)<1e-8):
        raise ValueError("Encoded movie differs from the declared 100-frame, 10-fps H.264 output")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--render-only",action="store_true")
    args = parser.parse_args()
    w = args.workspace.resolve()
    output = (args.output or w/"deliverables/recovered_example").resolve()
    output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter(); before = resources(w)
    # Conservative one-case budget; no host-RAM assumption or concurrency.
    if before["container_memory_available_bytes"] is not None and before["container_memory_available_bytes"] < 1024**3:
        raise RuntimeError("Less than 1 GiB measured container headroom; no reconstruction scheduled")
    reserve = max(WORKFLOW_DISK_RESERVE,2*1024**3,int(.2*before["disk_free_bytes"]))
    if before["disk_free_bytes"]-50*1024**2 < reserve:
        raise RuntimeError("The reconstruction would violate its disk reserve")
    adapter, train, window, fit, binding = verify(w)
    prediction_path = output/"recomputed_A_bayes_t00_a8000.npz"
    numerical_path = output/"reconstruction.json"
    verify_seconds = time.perf_counter()-start
    if args.render_only:
        numerical = json.loads(numerical_path.read_text())
        if (numerical["binding"] != binding or numerical["prediction_sha256"] != sha(prediction_path) or
                numerical["identity"] != content_hash({k:v for k,v in numerical.items() if k!="identity"})):
            raise ValueError("Stored reconstruction differs from its identity-bound numerical receipt")
        with np.load(prediction_path,allow_pickle=False) as saved:
            arrays = {key:saved[key] for key in saved.files}
    else:
        if prediction_path.exists() or numerical_path.exists():
            raise FileExistsError("Completed or partial reconstruction already exists; inspect it and use --render-only")
        started = time.perf_counter()
        samples = np.empty((PATHS,H,12),dtype=np.float64)
        batch_ranges = []
        for batch in fit.iter_forecasts(window.panel,H,window.target_coordinates,window.target_site_ids,
                                       PATHS,binding["forecast_stream"],batch_size=BATCH):
            a,b = batch.metadata["path_start"],batch.metadata["path_stop"]
            samples[a:b] = train.transform.inverse(batch.samples); batch_ranges.append([a,b])
        mean = samples.mean(axis=0,dtype=np.float64)
        forecast_seconds = time.perf_counter()-started
        if not np.isfinite(samples).all():
            raise FloatingPointError("Reconstructed predictive paths are not all finite")
        # Labels enter only now. The unchanged facade's internal disk-read limitation is recorded below.
        evaluation = adapter.evaluation_window(0,A,"test",history=HISTORY,horizon=H)
        started = time.perf_counter()
        metrics = score(mean,samples,evaluation.values,train.transform.scale,window.target_coordinates)
        arrays = {"truth":evaluation.values,"mean":mean,"x":window.target_x,"input_x":window.x,
                  "times":window.target_time,"site_ids":window.target_site_ids,
                  "quantile_levels":LEVELS,"quantiles":np.quantile(samples,LEVELS,axis=0),
                  "paths_first_32":samples[:32].copy()}
        lead = np.array(HISTORICAL["leads"])-1
        comparison = {"source":HISTORICAL["source"],"leads":HISTORICAL["leads"],
            "rmse_max_absolute_difference":float(np.max(np.abs(np.asarray(metrics["rmse_by_horizon"])[lead]-HISTORICAL["rmse"]))),
            "width90_max_absolute_difference":float(np.max(np.abs(np.asarray(metrics["intervals"]["0.9"]["width_by_horizon"])[lead]-HISTORICAL["width90"]))),
            "mean_abs_max_difference":float(abs(np.abs(mean).max()-HISTORICAL["mean_abs_max"])),
            "truth_abs_max_difference":float(abs(np.abs(evaluation.values).max()-HISTORICAL["truth_abs_max"]))}
        atomic_npz(prediction_path,**arrays)
        numerical = {"schema":"ks-recomputed-example/1","status":"recomputed_from_retained_fit",
            "created_utc":utc(),"binding":binding,"original_prediction_file_bytes_recovered":False,
            "new_fit_count":0,"new_selection_count":0,"new_forecast_origin_count":1,
            "scientific_scope":"one predeclared diagnostic example; no new main aggregate or ranking",
            "observation_access":"39 permitted input histories only; 12 disjoint target coordinates",
            "io_limitation":IO_LIMITATION,"units":"physical clean KS field",
            "static_draw_per_complete_path":True,"batch_ranges":batch_ranges,
            "all_complete_paths_content_sha256":content_hash(samples),
            "stored_paths":32,"scores":metrics,"historical_value_comparison":comparison,
            "prediction_sha256":sha(prediction_path),
            "forecast_seconds":forecast_seconds,"score_and_write_seconds":time.perf_counter()-started,
            "resource_after_forecast":resources(w)}
        numerical["identity"]=content_hash(numerical); atomic_json(numerical_path,numerical)
        print(json.dumps({"event":"numerical_reconstruction_complete","forecast_seconds":forecast_seconds,
                          "rmse":metrics["rmse"],"crps":metrics["crps"],"comparison":comparison}),flush=True)
        del samples
    del fit,adapter,train,window
    render_start = time.perf_counter(); movie = render(arrays,output)
    files = [prediction_path,numerical_path,output/"recomputed_A_bayes_t00_a8000_heatmap.png",
             output/"recomputed_A_bayes_t00_a8000_heatmap.pdf",output/"recomputed_A_bayes_t00_a8000.mp4",
             output/"recomputed_A_bayes_t00_a8000_intervals.png",output/"recomputed_A_bayes_t00_a8000_intervals.pdf"]
    receipt = {"schema":"ks-recomputed-example-artifacts/1","status":"rendered_pending_visual_review",
        "created_utc":utc(),"reconstruction_identity":numerical["identity"],
        "freeze_identity":FREEZE_ID,"fit_sha256":FIT_SHA,
        "recovery_script_sha256":sha(Path(__file__)),
        "files":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in files],
        "movie_probe":movie,"timing_seconds":{"verification":verify_seconds,"render":time.perf_counter()-render_start,
                                           "total":time.perf_counter()-start},
        "resource_before":before,"resource_after":resources(w),
        "task_memory_admission_bytes":1024**3,"disk_reserve_bytes":reserve,
        "workflow_initial_disk_reserve_bytes":WORKFLOW_DISK_RESERVE,
        "memory_measurement":"getrusage process high-water RSS and largest-child high-water RSS, separately; cgroup use includes other processes and file cache",
        "figure_scope":"12 target rows only; 39 input sites shown solely as a coordinate rug",
        "no_full_frame_dump":True,"render_only_reused_numerics":args.render_only}
    atomic_json(output/"artifact_receipt.json",receipt)
    print(json.dumps({"event":"artifacts_ready","output":str(output),"movie":movie,"resources":receipt["resource_after"]}),flush=True)


if __name__ == "__main__":
    main()

