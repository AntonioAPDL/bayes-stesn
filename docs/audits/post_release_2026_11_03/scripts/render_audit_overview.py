"""Render qualification and recorded cost from the saved audit evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


MODELS = [
    ("bayes_esn_ridge", "Bayesian ESN"),
    ("basis_lgssm", "Basis LGSSM"),
    ("separable_stgp", "Separable GP"),
    ("basis_gru", "Probabilistic basis GRU"),
    ("basis_krr", "Basis KRR"),
    ("eqesn_rollout", "E-QESN rollout"),
    ("dcrnn", "Diffusion GRU adaptation"),
]


def render(evidence: Path, output: Path) -> None:
    results = json.loads((evidence/"results_evidence.json").read_text())
    compute = json.loads((evidence/"compute_evidence.json").read_text())
    table = {row["model"]: row for row in results["pilot"]["model_table"]}
    costs = compute["pilot_compute_by_model"]
    for model, _ in MODELS:
        assert table[model]["attempts"] == costs[model]["jobs"]
        assert np.isclose(table[model]["sum_total_seconds_including_tuning"],
                          costs[model]["total_seconds_including_tuning"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#d4d9df",
        "text.color": "#172c42", "axes.labelcolor": "#172c42",
        "xtick.color": "#526273", "ytick.color": "#172c42",
        "svg.hashsalt": "bayes-stesn-post-release-audit-20261103",
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.6, 6.5),
                                gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.185, right=.965, top=.76, bottom=.20, wspace=.32)
    y = np.arange(len(MODELS))
    qualified = np.array([table[m]["qualified"]/table[m]["attempts"] for m, _ in MODELS])
    ax.barh(y, np.ones(len(y)), height=.56, color="#f1ded7")
    ax.barh(y, qualified, height=.56, color="#276a8a")
    ax.set_yticks(y, [label for _, label in MODELS])
    ax.set_xlim(0, 1.23)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Fraction meeting each model's declared gate", labelpad=12)
    ax.set_title("Fit reliability", loc="left", fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", color="#e4e8ec", linewidth=.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for j, (model, _) in enumerate(MODELS):
        row = table[model]
        ax.text(1.035, j, f'{row["qualified"]}/{row["attempts"]}', va="center",
                fontsize=11, fontweight="bold")
    tune = np.array([costs[m]["tuning_seconds"]/60 for m, _ in MODELS])
    total = np.array([costs[m]["total_seconds_including_tuning"]/60 for m, _ in MODELS])
    bx.barh(y, tune, height=.56, color="#276a8a", label="Tuning")
    bx.barh(y, total-tune, left=tune, height=.56, color="#afc6d0",
            label="Final fit, forecast and other job work")
    bx.set_yticks(y, [""]*len(y))
    bx.set_xlim(0, 114)
    bx.set_xticks([0, 25, 50, 75, 100])
    bx.set_xlabel("Summed job time including tuning (minutes)", labelpad=12)
    bx.set_title("Where the computation went", loc="left", fontsize=16,
                 fontweight="bold", pad=20)
    bx.tick_params(axis="both", length=0)
    bx.grid(axis="x", color="#e4e8ec", linewidth=.7)
    bx.set_axisbelow(True)
    bx.invert_yaxis()
    for j, value in enumerate(total):
        bx.text(value+1.3, j, f"{value:.2f}" if value < 1 else f"{value:.1f}",
                va="center", fontsize=11)
    bx.legend(loc="upper left", bbox_to_anchor=(-.015, -.20), frameon=False,
              fontsize=9.5, ncol=1)
    fig.text(.035, .94, "Improve reliability and tuning before enlarging the study",
             fontsize=21, fontweight="bold")
    fig.text(.035, .875, "Archived development pilot  |  20 independent datasets  |  300 model/task attempts",
             fontsize=12, color="#526273")
    fig.text(.035, .065,
             "Qualification criteria differ across families. Simple controls: 60/60 qualified.\n"
             "Time is summed across jobs; it is not campaign elapsed time. "
             "Source: original release receipts, independently re-aggregated.",
             fontsize=9.5, color="#526273", linespacing=1.6)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/"audit_overview.png", dpi=180, facecolor="white",
                metadata={"Title": "bayes-stesn post-release audit"})
    fig.savefig(output/"audit_overview.svg", facecolor="white",
                metadata={"Date": None, "Title": "bayes-stesn post-release audit"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.output)
    print(args.output/"audit_overview.png")


if __name__ == "__main__":
    main()
