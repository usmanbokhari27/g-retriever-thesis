"""Confidence-head calibration (reliability diagram + ECE) and qualitative
flip analysis for the refinement loop. Runs entirely off the two prediction
CSVs already pulled from EC2 -- no GPU.

Calibration target: whether the (level-0) retrieved subgraph contains a gold
answer entity -- the exact label the confidence head was trained on. We read
it back from the `desc` node list using the repo's own match() normaliser, and
score it against the `confidence` column of the no-refine run (level-0 subgraph,
level-0 confidence).
"""

# NOTE: paths below resolve against this submission bundle's results/
# directory. This is the only change from the script as run during the
# project, where these files sat in the working directory.

import re
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_PRED = _os.path.join(_RES, "test_predictions")
REFINE = _os.path.join(_PRED, "config5_refine.csv")
NOREFINE = _os.path.join(_PRED, "config5_control_norefine.csv")


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(c for c in s if c not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(hay: str, needle: str) -> bool:
    return normalize(needle) in normalize(hay)


def hit(pred, label):
    pred = str(pred).replace("|", "\n").split("\n")
    ans = str(label).split("|")
    pred_str = " ".join(pred)
    return int(any(match(pred_str, a) for a in ans))


def node_text(desc: str) -> str:
    """Concatenate the node_attr column of the desc block into one string."""
    lines = str(desc).splitlines()
    out = []
    for ln in lines:
        if "," not in ln or ln.startswith("node_id") or ln.startswith("src"):
            continue
        # node rows: "id,attr..."; stop once we reach the edge block
        first = ln.split(",", 1)
        if first[0].strip().isdigit() and len(first) > 1:
            out.append(first[1])
    return " ".join(out)


def subgraph_has_gold(desc, label):
    text = node_text(desc)
    return int(any(match(text, a) for a in str(label).split("|")))


on = pd.read_json(REFINE, lines=True).set_index("id").sort_index()
off = pd.read_json(NOREFINE, lines=True).set_index("id").sort_index()
common = on.index.intersection(off.index)
on, off = on.loc[common], off.loc[common]

# ---------- Calibration on the no-refine (level-0) run ----------
off = off.copy()
off["y"] = [subgraph_has_gold(d, l) for d, l in zip(off.desc, off.label)]
conf = off.confidence.to_numpy()
y = off.y.to_numpy()
print(f"calibration n={len(off)}  positive rate={y.mean():.4f}  "
      f"mean conf={conf.mean():.4f}")

# Reliability diagram: equal-width bins over observed confidence range.
n_bins = 10
edges = np.linspace(conf.min(), conf.max(), n_bins + 1)
bin_id = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
xs, ys, ws = [], [], []
ece = 0.0
for b in range(n_bins):
    m = bin_id == b
    if not m.any():
        continue
    c_mean, acc = conf[m].mean(), y[m].mean()
    xs.append(c_mean); ys.append(acc); ws.append(m.sum())
    ece += (m.sum() / len(conf)) * abs(acc - c_mean)
print(f"ECE={ece:.4f}  (10 equal-width bins)")

fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", lw=1, label="perfect calibration")
sizes = 30 + 220 * np.array(ws) / max(ws)
ax.scatter(xs, ys, s=sizes, color="#1f6feb", alpha=0.85, edgecolor="white",
           linewidth=0.8, zorder=3, label="confidence bins")
ax.plot(xs, ys, color="#1f6feb", lw=1.2, alpha=0.6, zorder=2)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Fraction of subgraphs containing a gold answer")
ax.set_xlim(0.35, 1.0); ax.set_ylim(0.35, 1.0)
ax.set_aspect("equal")
ax.grid(True, color="#e8eaed", lw=0.6)
ax.text(0.40, 0.94, f"AUC 0.662\nECE {ece:.3f}", fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d7de"))
ax.legend(loc="lower right", fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("confidence_calibration.png", dpi=200)
print("wrote confidence_calibration.png")

# ---------- Qualitative flip analysis on the fired subset ----------
fired = on.num_refine_iters > 0
on_f, off_f = on[fired], off.loc[on[fired].index]
rows = []
for i in on_f.index:
    h_on = hit(on_f.at[i, "pred"], on_f.at[i, "label"])
    h_off = hit(off_f.at[i, "pred"], off_f.at[i, "label"])
    rows.append(dict(
        id=i,
        question=re.sub(r"^Question:\s*|\s*Answer:\s*$", "",
                        str(on_f.at[i, "question"])).strip(),
        gold=on_f.at[i, "label"],
        pred_off=off_f.at[i, "pred"],
        pred_on=on_f.at[i, "pred"],
        conf_off=round(float(off_f.at[i, "confidence"]), 3),
        conf_on=round(float(on_f.at[i, "confidence"]), 3),
        iters=int(on_f.at[i, "num_refine_iters"]),
        flip=h_on - h_off,
    ))
flips = pd.DataFrame(rows)
w2r = flips[flips.flip > 0]
r2w = flips[flips.flip < 0]
print(f"\nfired={len(flips)}  wrong->right={len(w2r)}  right->wrong={len(r2w)}")
flips.to_csv("fired_flip_table.csv", index=False)
print("wrote fired_flip_table.csv")

pd.set_option("display.max_colwidth", 60)
print("\n=== WRONG -> RIGHT (refinement fixed these) ===")
for _, r in w2r.iterrows():
    print(f"[{r.id}] Q: {r.question}")
    print(f"    gold: {r.gold}")
    print(f"    off ({r.conf_off}): {r.pred_off}")
    print(f"    on  ({r.conf_on}, {r.iters} it): {r.pred_on}")
print("\n=== RIGHT -> WRONG (refinement broke these) ===")
for _, r in r2w.iterrows():
    print(f"[{r.id}] Q: {r.question}")
    print(f"    gold: {r.gold}")
    print(f"    off ({r.conf_off}): {r.pred_off}")
    print(f"    on  ({r.conf_on}, {r.iters} it): {r.pred_on}")
