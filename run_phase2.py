import json, sys, os, traceback

with open('daniel_phase_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

namespace = {}

def run_cell(src, idx):
    try:
        exec(src, namespace)
        return True, ""
    except Exception as e:
        return False, traceback.format_exc()

log_path = "results/phase2_run.log"
with open(log_path, "w", encoding="utf-8") as log:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        header = f"\n{'='*60}\nCELL {i}\n{'='*60}\n"
        log.write(header)
        log.flush()
        print(header, end="")
        sys.stdout.flush()
        ok, err = run_cell(src, i)
        if not ok:
            msg = f"ERROR in cell {i}:\n{err}\n"
            log.write(msg)
            print(msg)
        else:
            log.write("OK\n")
            print("OK")
        sys.stdout.flush()
        log.flush()

print("\n=== RUN COMPLETE ===")
print(f"Log saved to {log_path}")
