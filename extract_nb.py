import json

with open('daniel_phase_2_quantum_advantage_check.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code += "".join(cell['source']) + "\n\n"

with open('daniel_phase_2_quantum_advantage_check.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Extraction successful.")
