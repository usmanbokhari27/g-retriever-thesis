"""Fired-subset analysis for config #5 (iterative refinement).

Replicates src/utils/evaluate.py scoring exactly, then restricts the
refine-ON vs refine-OFF comparison to the samples where refinement
actually fired (num_refine_iters > 0). The whole-test-set delta dilutes
the mechanism's effect across the ~90% it never touches.
"""

# NOTE: paths below resolve against this submission bundle's results/
# directory. This is the only change from the script as run during the
# project, where these files sat in the working directory.

import re
import string
import pandas as pd

import os as _os
_RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")
_PRED = _os.path.join(_RES, "test_predictions")
REFINE = _os.path.join(_PRED, "config5_refine.csv")
NOREFINE = _os.path.join(_PRED, "config5_control_norefine.csv")


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    return sum(1.0 for a in answer if match(prediction, a)) / len(answer)


def eval_hit(prediction, answer):
    return int(any(match(prediction, a) for a in answer))


def score_rows(df):
    """Per-sample metrics, matching get_accuracy_webqsp's per-row logic."""
    out = []
    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):
        prediction = str(prediction).replace("|", "\n").split("\n")
        answer = str(answer).split("|")
        f1, prec, rec = eval_f1(prediction, answer)
        pred_str = " ".join(prediction)
        out.append(
            dict(hit=eval_hit(pred_str, answer), acc=eval_acc(pred_str, answer),
                 f1=f1, prec=prec, rec=rec)
        )
    return pd.DataFrame(out, index=df.index)


def summarize(name, scored):
    m = scored.mean() * 100
    print(f"{name:<34} n={len(scored):>5}  Hit {m.hit:6.2f}  Acc {m.acc:6.2f}  "
          f"Prec {m.prec:6.2f}  Rec {m.rec:6.2f}  F1 {m.f1:6.2f}")
    return m


on = pd.read_json(REFINE, lines=True).set_index("id").sort_index()
off = pd.read_json(NOREFINE, lines=True).set_index("id").sort_index()

# Align on the shared id set so ON and OFF are compared sample-for-sample.
common = on.index.intersection(off.index)
on, off = on.loc[common], off.loc[common]
print(f"aligned samples: {len(common)}  (ON {len(common)}, OFF {len(common)})\n")

on_s, off_s = score_rows(on), score_rows(off)

fired = on["num_refine_iters"] > 0
n_fired = int(fired.sum())
print(f"refinement fired on {n_fired}/{len(on)} = {100*n_fired/len(on):.2f}%")
print(f"mean iters {on.num_refine_iters.mean():.3f}, max {int(on.num_refine_iters.max())}")
print(f"iter histogram: {on.num_refine_iters.value_counts().sort_index().to_dict()}\n")

print("=== FULL TEST SET ===")
summarize("OFF (control)", off_s)
m_on_all = summarize("ON  (config #5)", on_s)

print("\n=== FIRED SUBSET (the mechanism's real effect) ===")
m_off_f = summarize("OFF on fired samples", off_s[fired.values])
m_on_f = summarize("ON  on fired samples", on_s[fired.values])
print("  delta " + "  ".join(
    f"{k} {m_on_f[k]-m_off_f[k]:+.2f}" for k in ["hit", "acc", "prec", "rec", "f1"]))

print("\n=== NOT-FIRED SUBSET (sanity: should be ~identical) ===")
m_off_n = summarize("OFF not fired", off_s[~fired.values])
m_on_n = summarize("ON  not fired", on_s[~fired.values])
print("  delta " + "  ".join(
    f"{k} {m_on_n[k]-m_off_n[k]:+.2f}" for k in ["hit", "acc", "prec", "rec", "f1"]))

# Did refinement flip individual samples, and in which direction?
flips = on_s.hit[fired.values] - off_s.hit[fired.values]
print(f"\nOn fired samples: {int((flips>0).sum())} wrong->right, "
      f"{int((flips<0).sum())} right->wrong, {int((flips==0).sum())} unchanged")

print(f"\nConfidence on fired {on.confidence[fired].mean():.3f} "
      f"vs not fired {on.confidence[~fired].mean():.3f}")
