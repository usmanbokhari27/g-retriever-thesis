"""Regenerate Figures 4 (confidence calibration) and 5 (refinement sweep)
for the thesis, at publication quality.

Figure 4 is recomputed from the raw week-5 prediction CSVs using the same
label definition as the original analysis (repo `match()` normaliser applied
to the node_attr column of the level-0 subgraph description), so the numbers
are identical to those quoted in the paper.

Figure 5 is built from week6_sweep/sweep_results.csv.

Both are written as PDF (vector, for LaTeX) and PNG (for preview), sized for
a 3.3 in single column in a two-column paper.

Run:  py make_figures.py
"""

# NOTE: paths below resolve against this submission bundle's results/
# directory. This is the only change from the script as run during the
# project, where these files sat in the working directory.

import os
import re
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
W5 = os.path.join(RESULTS, "test_predictions")
W6 = os.path.join(RESULTS, "validation_sweep")
FIGDIR = os.path.join(RESULTS, "figures")
NOREFINE = os.path.join(W5, "config5_control_norefine.csv")
SWEEP = os.path.join(W6, "sweep_results.csv")

# --------------------------------------------------------------------------
# Palette (validated: CVD dE 24.7, normal-vision dE 33.6, all >= 3:1 on white)
# --------------------------------------------------------------------------
BLUE = "#2a78d6"     # categorical slot 1  -> Hit@1 / calibration bins
ORANGE = "#eb6834"   # categorical slot 2  -> F1
BAR = "#9ec5f4"      # sequential blue 200 -> % refined bars
INK = "#0b0b0b"      # primary ink
INK2 = "#52514e"     # secondary ink
MUTED = "#898781"    # axis / tick labels
GRID = "#e1e0d9"     # hairline gridline
AXIS = "#c3c2b7"     # baseline / axis

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": AXIS,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def style(ax, grid_axis="both"):
    """Recessive chrome: hairline grid behind the data, no top/right spines."""
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf and {stem}.png")


# ==========================================================================
# Shared label logic (identical to week5_results/calib_and_flips.py)
# ==========================================================================
def normalize(s):
    s = str(s).lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(hay, needle):
    return normalize(needle) in normalize(hay)


def node_text(desc):
    out = []
    for ln in str(desc).splitlines():
        if "," not in ln or ln.startswith("node_id") or ln.startswith("src"):
            continue
        first = ln.split(",", 1)
        if first[0].strip().isdigit() and len(first) > 1:
            out.append(first[1])
    return " ".join(out)


def subgraph_has_gold(desc, label):
    text = node_text(desc)
    return int(any(match(text, a) for a in str(label).split("|")))


def roc_auc(y, score):
    """Rank-based AUC with tie correction (no sklearn dependency)."""
    y = np.asarray(y)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s_sorted = np.asarray(score)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n_pos, n_neg = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ==========================================================================
# FIGURE 4 -- confidence-head reliability diagram + score distribution
# ==========================================================================
def figure_calibration():
    off = pd.read_json(NOREFINE, lines=True).set_index("id").sort_index()
    y = np.array([subgraph_has_gold(d, l) for d, l in zip(off.desc, off.label)])
    conf = off.confidence.to_numpy()

    auc = roc_auc(y, conf)
    pos_rate = y.mean()

    n_bins = 10
    edges = np.linspace(conf.min(), conf.max(), n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    xs, ys, ws, ece = [], [], [], 0.0
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            continue
        c_mean, acc = conf[m].mean(), y[m].mean()
        xs.append(c_mean)
        ys.append(acc)
        ws.append(int(m.sum()))
        ece += (m.sum() / len(conf)) * abs(acc - c_mean)
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)

    print(f"  n={len(conf)}  positive rate={pos_rate:.4f}  "
          f"AUC={auc:.4f}  ECE={ece:.4f}")
    print(f"  mean conf | positive={conf[y == 1].mean():.4f}  "
          f"negative={conf[y == 0].mean():.4f}")
    print(f"  confidence range {conf.min():.3f}-{conf.max():.3f}")
    print("  bin   mean_conf  observed   gap   n")
    for k, (a, b, w) in enumerate(zip(xs, ys, ws)):
        print(f"   {k}     {a:.3f}      {b:.3f}   {b - a:+.3f}  {w:4d}")

    lo, hi = 0.35, 1.0
    fig = plt.figure(figsize=(3.3, 3.5))
    gs = GridSpec(2, 1, height_ratios=[3.1, 1.0], hspace=0.10, figure=fig)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    # --- reliability panel ---
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), color=MUTED, lw=0.9,
            zorder=2, label="perfect calibration")
    # operating point of the refinement loop
    ax.axvline(0.5, color=ORANGE, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    ax.annotate(r"$\tau = 0.5$", xy=(0.5, hi), xytext=(0.515, 0.965),
                color=ORANGE, fontsize=7, ha="left", va="top")

    ax.plot(xs, ys, color=BLUE, lw=1.1, alpha=0.55, zorder=3)
    ax.scatter(xs, ys, s=18 + 150 * ws / ws.max(), color=BLUE, alpha=0.9,
               edgecolor="white", linewidth=0.7, zorder=4,
               label="confidence bin (area $\\propto$ count)")

    ax.set_ylabel("Observed fraction with\ngold answer in subgraph",
                  color=INK2, linespacing=1.35)
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(lo, ys.min() - 0.04), hi)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelbottom=False)
    style(ax)

    ax.text(0.375, 0.965, f"AUC {auc:.3f}\nECE {ece:.3f}", fontsize=7.5,
            va="top", ha="left", color=INK, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=GRID, lw=0.6))
    leg = ax.legend(loc="lower right", frameon=False, handlelength=1.6,
                    borderpad=0.2, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(INK2)

    # --- score-distribution panel ---
    axh.hist(conf, bins=np.linspace(lo, hi, 40), color=BAR,
             edgecolor="white", linewidth=0.3, zorder=3)
    axh.axvline(0.5, color=ORANGE, lw=0.9, ls=(0, (1.5, 1.5)), zorder=4)
    axh.set_xlabel("Predicted confidence $c$", color=INK2)
    axh.set_ylabel("Questions", color=INK2)
    axh.set_xlim(lo, hi)
    axh.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    style(axh, grid_axis="y")

    save(fig, "confidence_calibration")
    return ece, auc


# ==========================================================================
# FIGURE 5 -- refinement threshold sweep (two panels, single scale each)
# ==========================================================================
def figure_sweep():
    df = pd.read_csv(SWEEP)
    ctrl = df[df.tag == "off"].iloc[0]
    s = df[(df.K == 2) & (df.tag != "off")].sort_values("tau")
    tau = s.tau.to_numpy()

    fig = plt.figure(figsize=(3.3, 3.5))
    gs = GridSpec(2, 1, height_ratios=[2.15, 1.0], hspace=0.12, figure=fig)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)

    # --- metric panel: one scale, both series are percentages ---
    ax.axhline(ctrl.hit, color=BLUE, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    ax.axhline(ctrl.f1, color=ORANGE, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
    ax.annotate(f"control {ctrl.hit:.1f}", xy=(0.695, ctrl.hit),
                xytext=(0.702, ctrl.hit + 0.4), color=BLUE, fontsize=6.5,
                ha="right", va="bottom")
    ax.annotate(f"control {ctrl.f1:.1f}", xy=(0.695, ctrl.f1),
                xytext=(0.702, ctrl.f1 + 0.4), color=ORANGE, fontsize=6.5,
                ha="right", va="bottom")

    ax.plot(tau, s.hit, "-o", color=BLUE, lw=1.4, ms=4.5,
            mec="white", mew=0.7, zorder=4, label="Hit@1")
    ax.plot(tau, s.f1, "--s", color=ORANGE, lw=1.4, ms=4.0,
            mec="white", mew=0.7, zorder=4, label="F1")

    ax.set_ylabel("Validation metric (%)", color=INK2)
    ax.set_ylim(45, 73.5)
    ax.set_yticks([45, 50, 55, 60, 65, 70])
    ax.tick_params(labelbottom=False)
    style(ax, grid_axis="y")
    leg = ax.legend(loc="center left", frameon=False, handlelength=2.0,
                    borderpad=0.2, labelspacing=0.35,
                    bbox_to_anchor=(0.015, 0.42))
    for t in leg.get_texts():
        t.set_color(INK2)

    # --- firing-rate panel ---
    axb.bar(tau, s.fired_pct, width=0.055, color=BAR, edgecolor="white",
            linewidth=0.5, zorder=3)
    for x, v in zip(tau, s.fired_pct):
        axb.annotate(f"{v:.0f}", xy=(x, v), xytext=(0, 1.6),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, color=INK2)
    axb.set_xlabel(r"Refinement threshold $\tau$   ($K=2$)", color=INK2)
    axb.set_ylabel("Questions refined (%)", color=INK2)
    axb.set_xlim(0.355, 0.745)
    axb.set_xticks([0.4, 0.5, 0.6, 0.7])
    axb.set_ylim(0, 72)
    axb.set_yticks([0, 25, 50])
    style(axb, grid_axis="y")

    save(fig, "refine_sweep")


if __name__ == "__main__":
    print("Figure 4: confidence calibration")
    figure_calibration()
    print("Figure 5: refinement sweep")
    figure_sweep()
