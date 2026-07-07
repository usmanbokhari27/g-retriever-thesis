"""Cleanup (Session-10 TODO): re-score saved WebQSP predictions after truncating
each one at the first end-of-sequence marker.

Background: under the LLaMA-3.1 tokenizer the training EOS string (``</s>``, a
LLaMA-2 chat marker) is *literal text*, not a special token, so
``skip_special_tokens=True`` never stripped it. The model emits
``<answer> </s> <hallucinated continuation>`` and the trailing text inflates the
predicted-entity count, depressing precision/F1 for EVERY config (~89% of preds
affected). Cutting each prediction at the first EOS marker removes that noise
WITHOUT re-running any GPU job — it just re-scores the predictions we already have.

This reuses the repo's OWN metric (``src.utils.evaluate.get_accuracy_webqsp``) on a
truncated copy, so the numbers are directly comparable to the reported ones — no
re-implemented metric to diverge. New runs through ``AdaptiveGraphLLM`` already
apply the same truncation at inference time (see ``adaptive_graph_llm.py``); this
script is only for the config #1/#2/#3 CSVs that were saved before the fix.

Run from the repo root (so ``src`` imports), e.g.:
    PYTHONPATH=. python truncate_eval.py output/webqsp/<...>_seed0.csv
    # or point it at several CSVs at once:
    PYTHONPATH=. python truncate_eval.py output/webqsp/*.csv
Writes a ``*_trunc.csv`` next to each input and prints before/after metrics.
"""
import json
import sys

# Same markers, same precedence as AdaptiveGraphLLM.EOS_MARKERS.
EOS_MARKERS = ('</s>', '<|eot_id|>', '<|end_of_text|>')


def _truncate(text):
    cut = len(text)
    for marker in EOS_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _load_rows(path):
    """The saved files are JSON-lines despite the .csv extension (schema:
    id, pred, label, question, desc)."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_rows(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def process(path):
    from src.utils.evaluate import get_accuracy_webqsp  # repo's own metric

    rows = _load_rows(path)
    affected = sum(1 for r in rows
                   if any(m in r.get('pred', '') for m in EOS_MARKERS))
    print(f'\n=== {path} ===')
    print(f'{len(rows)} predictions; {affected} '
          f'({affected / max(len(rows), 1):.1%}) contain an EOS marker.')

    out_path = path.rsplit('.', 1)[0] + '_trunc.csv'
    trunc = [dict(r, pred=_truncate(r.get('pred', ''))) for r in rows]
    _write_rows(out_path, trunc)

    # get_accuracy_webqsp prints the full metric block (Acc/Hit/Precision/Recall/F1)
    # and returns Hit@1. Call it on both files for a clean before/after.
    print('--- BEFORE truncation ---')
    before = get_accuracy_webqsp(path)
    print('--- AFTER truncation ----')
    after = get_accuracy_webqsp(out_path)
    print(f'Hit@1: {before:.4f} -> {after:.4f}  (Δ {after - before:+.4f})')
    print(f'wrote truncated copy: {out_path}')


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit('usage: PYTHONPATH=. python truncate_eval.py <pred.csv> [more.csv ...]')
    for p in paths:
        process(p)
