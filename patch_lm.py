import re, pathlib, shutil

p = pathlib.Path('src/utils/lm_modeling.py')
shutil.copy(p, p.with_suffix('.py.bak'))
s = p.read_text()

# 1) Wrap the SBERT forward in fp16 autocast, store result back as fp32
if 'torch.cuda.amp.autocast' in s:
    print('autocast already present -- skipping wrap')
else:
    def repl(m):
        indent, call = m.group(1), m.group(2)
        return (f'{indent}with torch.cuda.amp.autocast(dtype=torch.float16):\n'
                f'{indent}    {call}.float()')
    s, n = re.subn(r'( *)(embeddings = model\([^\n]*?\))\s*$', repl, s, count=1, flags=re.M)
    print(f'autocast wrap applied to {n} site (expected 1)')
    assert n == 1, 'forward call not matched -- aborting'

# 2) Bump batch size back up (fp16 halves memory)
s, n2 = re.subn(r'(?m)^batch_size\s*=\s*\d+', 'batch_size = 1024', s)
print(f'batch_size set to 1024: {n2} line(s)')

p.write_text(s)
print('OK -- lm_modeling.py patched (backup at src/utils/lm_modeling.py.bak)')
