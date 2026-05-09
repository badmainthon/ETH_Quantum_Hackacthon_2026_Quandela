import json

with open('daniel_phase_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find conclusion code cell
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'PHASE 2 AUTO-GENERATED CONCLUSION' in ''.join(c['source']):
        src = ''.join(c['source'])
        # Add QEA trade-off section before the final "="*70
        old_end = 'print("="*70)'
        new_section = '''# QEA-inspired cost-accuracy trade-off
print("\n=== QEA-Inspired Cost-Accuracy Trade-off ===")
if len(qea_rows) > 0:
    df_qea = pd.DataFrame(qea_rows)
    best_classical = df_qea[df_qea['model'].str.contains('Classical')].loc[df_qea['cost_per_accuracy'].idxmin()]
    merlin_aux_row = df_qea[df_qea['model'] == 'MerLin Aux (depth=1)']
    if len(merlin_aux_row) > 0:
        ma = merlin_aux_row.iloc[0]
        print(f"Classical best cost-per-accuracy: {best_classical['cost_per_accuracy']:.4f} ({best_classical['model']})")
        print(f"MerLin Aux cost-per-accuracy:     {ma['cost_per_accuracy']:.4f}")
        print(f"MerLin Aux accuracy (rel_l2):     {ma['rel_l2']:.4e}")
        print(f"Classical best accuracy (rel_l2): {best_classical['rel_l2']:.4e}")
        if ma['rel_l2'] > best_classical['rel_l2'] and ma['cost_per_accuracy'] > best_classical['cost_per_accuracy']:
            verdict = "The MerLin QPINN is WORSE in BOTH accuracy and cost (Option 4)."
        elif ma['rel_l2'] > best_classical['rel_l2'] and ma['cost_per_accuracy'] <= best_classical['cost_per_accuracy']:
            verdict = "The MerLin QPINN is less accurate but cheaper (Option 3-ish, but accuracy is worse)."
        elif ma['rel_l2'] <= best_classical['rel_l2'] and ma['cost_per_accuracy'] > best_classical['cost_per_accuracy']:
            verdict = "The MerLin QPINN is more accurate but more expensive (Option 1)."
        elif ma['rel_l2'] <= best_classical['rel_l2'] and ma['cost_per_accuracy'] <= best_classical['cost_per_accuracy']:
            verdict = "The MerLin QPINN is similarly accurate with lower resource proxy (Option 3)."
        else:
            verdict = "Inconclusive (Option 5)."
        print(f"\\nVerdict: {verdict}")
        print("\\nImportant: This is a simulator-based energy-proxy analysis, not a full-stack physical energy measurement.")
        print("No credible QEA (Quantum Energetic Advantage) is claimed.")

print("="*70)'''
        
        if old_end in src:
            src = src.replace(old_end, new_section)
            nb['cells'][i]['source'] = src.splitlines(keepends=True)
            print('Updated conclusion cell')
        else:
            print('Pattern not found')
        break

with open('daniel_phase_2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
