import json

with open('daniel_phase_2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the index of the "Final Conclusion" markdown cell
conclusion_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and 'Final Conclusion' in ''.join(c['source']):
        conclusion_idx = i
        break

if conclusion_idx is None:
    conclusion_idx = len(cells)

# New cell 1: QEA markdown
qea_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## QEA-Inspired Resource Analysis\n",
        "\n",
        "Honest cost-to-accuracy comparison including photonic resource proxies.\n",
        "\n",
        "**Cost-per-accuracy** = runtime_or_energy_proxy × relative_L2_error. Lower is better.\n",
        "\n",
        "**Energy proxy** (photonic only): approx_mzi_count × 0.5 mW + approx_phase_shifters × 2.56 mW per π-shift.\n",
        "This is a *proxy only* — not a full-stack physical energy measurement.\n"
    ]
}

# New cell 2: QEA code
qea_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import tracemalloc\n",
        "\n",
        "def measure_peak_memory_during_train(model_builder, config, mode):\n",
        "    \"\"\"Train a model and return peak memory in MB (Python-level via tracemalloc).\"\"\"\n",
        "    torch.manual_seed(0); np.random.seed(0); random.seed(0)\n",
        "    model = model_builder()\n",
        "    for p in model.parameters():\n",
        "        if p.is_floating_point(): p.data = p.data.to(DTYPE)\n",
        "    tracemalloc.start()\n",
        "    hist, elapsed = train_model(model, config, mode=mode)\n",
        "    current, peak = tracemalloc.get_traced_memory()\n",
        "    tracemalloc.stop()\n",
        "    return peak / (1024 * 1024), elapsed, model\n",
        "\n",
        "def count_circuit_components(circ):\n",
        "    \"\"\"Count BS, PS, and other components from circuit description.\"\"\"\n",
        "    desc = circ.describe()\n",
        "    bs_count = desc.count('BS.')\n",
        "    ps_count = desc.count('PS(')\n",
        "    mzi_approx = bs_count // 2  # each MZI ~ 2 BS in Reck decomposition\n",
        "    return {\n",
        "        'bs_count': bs_count,\n",
        "        'ps_count': ps_count,\n",
        "        'mzi_approx': mzi_approx,\n",
        "        'total_components': circ.ncomponents(),\n",
        "    }\n",
        "\n",
        "def extract_merlin_resources(model):\n",
        "    ql = None\n",
        "    for m in model.modules():\n",
        "        if hasattr(m, 'quantum_layer'):\n",
        "            ql = m.quantum_layer; break\n",
        "    if ql is None and hasattr(model, 'quantum'):\n",
        "        qmod = model.quantum\n",
        "        if hasattr(qmod, 'quantum_layer'): ql = qmod.quantum_layer\n",
        "    if ql is None:\n",
        "        return {}\n",
        "    circ = getattr(ql, 'circuit', None)\n",
        "    if circ is not None:\n",
        "        n_modes = getattr(circ, 'm', getattr(ql, 'input_size', 'unknown'))\n",
        "        depths = circ.depths()\n",
        "        ncomp = circ.ncomponents()\n",
        "        comp = count_circuit_components(circ)\n",
        "    else:\n",
        "        n_modes = getattr(ql, 'input_size', 'unknown')\n",
        "        depths = 'simple'\n",
        "        ncomp = 'unknown'\n",
        "        comp = {'bs_count': 'unknown', 'ps_count': 'unknown', 'mzi_approx': 'unknown', 'total_components': 'unknown'}\n",
        "    q_params = sum(p.numel() for p in ql.parameters() if p.requires_grad)\n",
        "    # cutoff / Hilbert space dimension from output_size\n",
        "    hilbert_dim = getattr(ql, 'output_size', 'unknown')\n",
        "    return {\n",
        "        'qumodes': n_modes,\n",
        "        'depth': depths,\n",
        "        'q_params': q_params,\n",
        "        'hilbert_dim': hilbert_dim,\n",
        "        'ncomp': ncomp,\n",
        "        **comp,\n",
        "    }\n",
        "\n",
        "# Re-run each baseline with memory measurement\n",
        "qea_cfg = TrainConfig(epochs=BASE_EPOCHS, lr=1e-2, print_every=0)\n",
        "qea_rows = []\n",
        "\n",
        "for name, builder, mode in [\n",
        "    ('Classical Direct (matched)', lambda: ClassicalDirectPINN(hidden=classical_hidden, depth=classical_depth), 'direct'),\n",
        "    ('Classical Aux (matched)', lambda: ClassicalAuxPINN(hidden=aux_hidden, depth=aux_depth), 'aux'),\n",
        "    ('MerLin Aux (depth=1)', lambda: MerlinAuxQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=1), 'merlin_aux'),\n",
        "    ('MerLin Direct (depth=1)', lambda: MerlinDirectQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=1), 'merlin_direct'),\n",
        "    ('Classical Direct (strong)', lambda: ClassicalDirectPINN(hidden=strong_hidden, depth=strong_depth), 'direct'),\n",
        "]:\n",
        "    print(f'\\nMeasuring {name}...')\n",
        "    peak_mb, elapsed, model = measure_peak_memory_during_train(builder, qea_cfg, mode)\n",
        "    metrics = evaluate_model(model, mode=mode)\n",
        "    row = {\n",
        "        'model': name,\n",
        "        'rel_l2': metrics['rel_l2'],\n",
        "        'pde_residual': metrics['pde_mse'],\n",
        "        'runtime_s': elapsed,\n",
        "        'peak_memory_mb': peak_mb,\n",
        "        'params': count_parameters(model),\n",
        "        'qumodes': 0,\n",
        "        'depth': 'N/A',\n",
        "        'cutoff': 'N/A',\n",
        "        'approx_gate_count': 'N/A',\n",
        "        'approx_mzi_count': 'N/A',\n",
        "        'approx_phase_shifters': 'N/A',\n",
        "        'energy_proxy_mw': 0.0,\n",
        "        'cost_per_accuracy': elapsed * metrics['rel_l2'],\n",
        "        'notes': '',\n",
        "    }\n",
        "    if 'merlin' in name.lower():\n",
        "        res = extract_merlin_resources(model)\n",
        "        row['qumodes'] = res.get('qumodes', 'unknown')\n",
        "        row['depth'] = str(res.get('depth', 'unknown'))\n",
        "        row['cutoff'] = res.get('hilbert_dim', 'unknown')\n",
        "        row['approx_gate_count'] = res.get('total_components', 'unknown')\n",
        "        row['approx_mzi_count'] = res.get('mzi_approx', 'unknown')\n",
        "        row['approx_phase_shifters'] = res.get('ps_count', 'unknown')\n",
        "        # Energy proxy: MZI ~ 0.5 mW each, PS ~ 2.56 mW per pi-shift (proxy only)\n",
        "        mzi_cnt = res.get('mzi_approx', 0) if isinstance(res.get('mzi_approx'), int) else 0\n",
        "        ps_cnt = res.get('ps_count', 0) if isinstance(res.get('ps_count'), int) else 0\n",
        "        energy = mzi_cnt * 0.5 + ps_cnt * 2.56\n",
        "        row['energy_proxy_mw'] = energy\n",
        "        row['cost_per_accuracy'] = energy * metrics['rel_l2']\n",
        "        row['notes'] = f\"Photonic proxy: {res.get('bs_count', '?')} BS, {ps_cnt} PS, {res.get('total_components', '?')} total comps\"\n",
        "    else:\n",
        "        row['notes'] = 'Classical MLP baseline'\n",
        "    qea_rows.append(row)\n",
        "    print(f\"  rel_l2={metrics['rel_l2']:.4e} | time={elapsed:.1f}s | mem={peak_mb:.1f}MB\")\n",
        "\n",
        "df_qea = pd.DataFrame(qea_rows)\n",
        "print('\\n=== QEA-Inspired Resource Table ===')\n",
        "print(df_qea.to_string(index=False))\n"
    ]
}

# Insert before conclusion
cells.insert(conclusion_idx, qea_md)
cells.insert(conclusion_idx + 1, qea_code)

with open('daniel_phase_2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Added QEA cells before conclusion.')
