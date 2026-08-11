"""Aggregate the Week-6 validation sweep into a sensitivity table + figure.

Run locally after pulling the per-combo CSVs
(output/webqsp_attn/..._seed0_val_<tag>.csv) into this folder. Scores each combo
with the repo's exact metric functions, picks the best (tau, K) by validation
Hit@1 (F1 tie-break), and writes:
  - sweep_results.csv        machine-readable per-combo metrics
  - sweep_table.tex          LaTeX table body for the thesis
  - refine_sweep.png         tau sensitivity (Hit@1 + F1 + fire-rate) at K=2

Nothing here touches the test set; this only tunes hyperparameters on val.
"""

# NOTE: paths below resolve against this submission bundle's results/
# directory. This is the only change from the script as run during the
# project, where these files sat in the working directory.

import re
import glob
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_SWEEP = _os.path.join(_RES, "validation_sweep")
GLOB = _os.path.join(_SWEEP, "predictions", "val_*.csv")


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(hay, needle):
    return normalize(needle) in normalize(hay)


def score(df):
    hit = acc = f1 = prec = rec = 0.0
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        pred = str(pred).replace("|", "\n").split("\n")
        ans = str(label).split("|")
        ps = " ".join(pred)
        matched = sum(1 for a in ans if match(ps, a))
        p = matched / len(pred) if pred else 0.0
        r = matched / len(ans) if ans else 0.0
        f1 += (2 * p * r / (p + r)) if (p + r) else 0.0
        prec += p; rec += r
        acc += sum(1 for a in ans if match(ps, a)) / len(ans)
        hit += int(any(match(ps, a) for a in ans))
    n = len(df)
    return dict(hit=100 * hit / n, acc=100 * acc / n, prec=100 * prec / n,
                rec=100 * rec / n, f1=100 * f1 / n, n=n)


def parse_tag(tag):
    if tag == "off":
        return dict(refine=False, tau=None, K=0)
    m = re.match(r"tau([\d.]+)_K(\d+)", tag)
    return dict(refine=True, tau=float(m.group(1)), K=int(m.group(2)))


rows = []
for path in sorted(glob.glob(GLOB)):
    tag = re.search(r"^val_(.+)\.csv$", _os.path.basename(path)).group(1)
    df = pd.read_json(path, lines=True)
    meta = parse_tag(tag)
    s = score(df)
    fired = int((df.num_refine_iters > 0).sum()) if "num_refine_iters" in df else 0
    mean_it = float(df.num_refine_iters.mean()) if "num_refine_iters" in df else 0.0
    rows.append(dict(tag=tag, **meta, **s,
                     fired=fired, fired_pct=100 * fired / s["n"],
                     mean_iters=mean_it))

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit(f"No CSVs matched {GLOB} in this folder.")
res = res.sort_values(["refine", "K", "tau"], na_position="first").reset_index(drop=True)
res.to_csv(_os.path.join(_SWEEP, "sweep_results.csv"), index=False)
pd.set_option("display.width", 160)
print(res.to_string(index=False,
                    columns=["tag", "tau", "K", "hit", "f1", "prec",
                             "fired_pct", "mean_iters"]))

# Best refining combo by val Hit@1, then F1.
ref = res[res.refine].copy()
best = ref.sort_values(["hit", "f1"], ascending=False).iloc[0]
off_hit = res.loc[~res.refine, "hit"]
print(f"\nBEST (val): tau={best.tau} K={int(best.K)} -> "
      f"Hit {best.hit:.2f}  F1 {best.f1:.2f}  fired {best.fired_pct:.1f}%")
if len(off_hit):
    print(f"OFF control (val): Hit {off_hit.iloc[0]:.2f}  "
          f"(delta {best.hit - off_hit.iloc[0]:+.2f})")

# ---- LaTeX table body ----
lines = []
off = res[~res.refine]
if len(off):
    o = off.iloc[0]
    lines.append(f"Refine off (control) & -- & -- & {o.hit:.2f} & {o.f1:.2f} & -- \\\\")
lines.append(r"\midrule")
for _, r in ref.sort_values(["K", "tau"]).iterrows():
    star = r"\textbf" if r.tag == best.tag else ""
    hit = f"{star}{{{r.hit:.2f}}}" if star else f"{r.hit:.2f}"
    lines.append(f"{r.tau:.1f} & {int(r.K)} & {r.fired_pct:.1f}\\% & "
                 f"{hit} & {r.f1:.2f} & {r.mean_iters:.3f} \\\\")
with open("sweep_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote sweep_table.tex, sweep_results.csv")

# ---- Figure: tau sensitivity at K=2 ----
k2 = ref[ref.K == 2].sort_values("tau")
if len(k2) >= 2:
    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    ax1.plot(k2.tau, k2.hit, "o-", color="#1f6feb", label="Hit@1")
    ax1.plot(k2.tau, k2.f1, "s--", color="#8250df", label="F1")
    if len(off):
        ax1.axhline(off.iloc[0].hit, color="#9aa0a6", ls=":", lw=1,
                    label="Hit@1 (refine off)")
    ax1.set_xlabel(r"Refinement threshold $\tau$ (K=2)")
    ax1.set_ylabel("Validation metric (%)")
    ax2 = ax1.twinx()
    ax2.bar(k2.tau, k2.fired_pct, width=0.03, color="#1f6feb", alpha=0.15)
    ax2.set_ylabel("% questions refined", color="#57606a")
    ax2.set_ylim(0, max(k2.fired_pct.max() * 1.6, 5))
    ax1.legend(loc="lower left", fontsize=8, frameon=False)
    for s in ("top",):
        ax1.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("refine_sweep.png", dpi=200)
    print("wrote refine_sweep.png")
