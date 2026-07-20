"""Idempotently add the Week-5 refinement argparse flags to src/config.py.

Adding flags to config.py is the sanctioned extension pattern (handover §4.5).
Backed up to src/config.py.bak_refine on first run. Inserts right before
``args = parser.parse_args()``, matching indentation.

Run once from the repo root:
    python patch_refine_flags.py
"""
import shutil

P = 'src/config.py'
FLAGS = [
    'parser.add_argument("--refine", action="store_true")',
    'parser.add_argument("--refine_tau", type=float, default=0.5)',
    'parser.add_argument("--refine_max_iters", type=int, default=2)',
    'parser.add_argument("--refine_max_nodes", type=int, default=60)',
    'parser.add_argument("--checkpoint_path", type=str, default="")',
]


def main():
    src = open(P).read()
    if '--refine' in src and '--refine_max_nodes' in src:
        print('config.py: Week-5 flags already present.')
        return
    shutil.copyfile(P, P + '.bak_refine')
    out, inserted = [], False
    for ln in src.splitlines(keepends=True):
        if not inserted and 'parser.parse_args(' in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            for flag in FLAGS:
                out.append(f'{indent}{flag}\n')
            inserted = True
        out.append(ln)
    if not inserted:
        raise SystemExit('ERROR: could not find parser.parse_args() in config.py '
                         '— add the five flags manually.')
    open(P, 'w').write(''.join(out))
    print('config.py: Week-5 flags inserted (backup at src/config.py.bak_refine).')


if __name__ == '__main__':
    main()
