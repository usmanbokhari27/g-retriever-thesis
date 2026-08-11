"""Idempotently add the two Week-4 argparse flags to src/config.py.
Run from the repo root: python ~/patch_config_flags.py
Sanctioned extension pattern (handover 4.5); backs up to config.py.bak_confidence.
"""
p = 'src/config.py'
src = open(p).read()
if '--confidence_head' in src:
    print('config.py: flags already present.')
else:
    import shutil
    shutil.copyfile(p, p + '.bak_confidence')
    out, ins = [], False
    for ln in src.splitlines(keepends=True):
        if not ins and 'parser.parse_args(' in ln:
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append(ind + 'parser.add_argument("--confidence_head", action="store_true")\n')
            out.append(ind + 'parser.add_argument("--confidence_weight", type=float, default=0.1)\n')
            ins = True
        out.append(ln)
    if not ins:
        raise SystemExit('ERROR: could not find parser.parse_args() in config.py')
    open(p, 'w').write(''.join(out))
    print('config.py: flags inserted (backup at src/config.py.bak_confidence).')
