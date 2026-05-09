import json
with open('daniel_phase_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:60].replace('\n',' ')
    print(f'{i:2d} {c["cell_type"]} | {src}')
