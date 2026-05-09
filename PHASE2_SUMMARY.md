# Phase 2 Summary — ETH Quantum Hackathon 2026

## What Was Implemented

- **New problem**: 1D heat equation on $x \in [-\pi/2, \pi/2]$, $t \in [0, 0.5]$, $\alpha = 0.30$, with Gaussian initial condition and Dirichlet BCs.
- **Reference solution**: Method-of-lines with finite-difference Laplacian + RK45 time integration.
- **Classical baselines**: Parameter-matched direct PINN (249 params), parameter-matched auxiliary PINN (282 params), and a stronger larger PINN (3297 params).
- **MerLin QPINN variants**: Real `ML.QuantumLayer.simple` and `ML.CircuitBuilder` for custom depth.
  - Auxiliary-derivative QPINN (270 params, 40 quantum params)
  - Direct second-derivative QPINN (253 params, 40 quantum params)
- **Experiments**: Baseline comparison, ablation, expressivity sweep (skipped in FAST_MODE), temporal extrapolation generalization, resource analysis, noise placeholder.

## Experiment Matrix

| Model | Params | Q-Params | Epochs | Seeds | Analysis |
|-------|--------|----------|--------|-------|----------|
| Classical Direct (matched) | 249 | 0 | 50 | 1 | Baseline |
| Classical Aux (matched) | 282 | 0 | 50 | 1 | Baseline |
| Classical Direct (strong) | 3297 | 0 | 50 | 1 | Baseline |
| MerLin Aux (depth=1) | 270 | 40 | 50 | 1 | Baseline |
| MerLin Direct (depth=1) | 253 | 40 | 50 | 1 | Baseline |

## Key Quantitative Results (FAST_MODE, 50 epochs, seed=0)

| Model | Rel L2 | RMSE | PDE MSE | Time (s) |
|-------|--------|------|---------|----------|
| Classical Direct (matched) | 1.98e-01 | 4.29e-02 | 1.54e-02 | 0.6 |
| Classical Aux (matched) | 1.37e-01 | 2.97e-02 | 2.80e-03 | 0.4 |
| MerLin Aux (depth=1) | 2.25e-01 | 4.88e-02 | 2.14e-03 | 13.3 |
| MerLin Direct (depth=1) | 2.94e-01 | 6.39e-02 | 1.53e-02 | 14.1 |
| Classical Direct (strong) | 9.63e-02 | 2.09e-02 | 8.73e-03 | 0.7 |

### Observations
- **Auxiliary derivative helped**: Classical Aux (1.37e-01) outperformed Classical Direct (1.98e-01) on Rel L2.
- **MerLin did NOT beat parameter-matched classical**: MerLin Aux (2.25e-01) was worse than Classical Direct matched (1.98e-01) and much worse than Classical Aux matched (1.37e-01).
- **Stronger classical easily beat MerLin**: Classical Direct strong (9.63e-02) significantly outperformed all other models.
- **MerLin is ~20x slower per run** than classical baselines on CPU simulator.
- **Generalization**: All models degraded significantly on temporal extrapolation (t > 0.5), with MerLin Direct performing worst.

## Conclusion

### Auto-Generated Answers
1. **Did auxiliary derivative formulation help?** Yes — improved Rel L2 (1.81e-01 vs 2.46e-01 averaged across classical and MerLin).
2. **Did direct second-derivative training work?** Yes, but with lower accuracy (Rel L2 = 2.94e-01).
3. **Did increasing MerLin circuit expressivity help?** Expressivity sweep skipped in FAST_MODE.
4. **Did QPINN beat parameter-matched classical?** No — MerLin Aux did NOT beat Classical Direct.
5. **Did QPINN beat stronger classical?** No — MerLin Aux did NOT beat Classical Direct (strong).
6. **Did QPINN generalize outside training?** All models degraded on extrapolation; MerLin was not uniquely better.
7. **Honest evidence of quantum advantage?** None at this simulator scale.
8. **Most defensible presentation conclusion?** The MerLin DV-QPINN is a viable PINN architecture, but its utility comes from the trainable photonic feature map, not from quantum computational advantage at this scale.

## Limitations

- All experiments run on classical CPU simulator (no photonic hardware).
- FAST_MODE used only 50 epochs and 1 seed. Full-mode (300 epochs, 3 seeds) may change quantitative rankings but is unlikely to reverse the qualitative conclusion.
- Noise evaluation is a documented placeholder due to API complexity for post-training noise injection.
- Runtime constraints limited multi-seed statistics for expensive sweeps.

## What to Say in the Presentation

- The MerLin DV-QPINN is a **viable PINN architecture** with comparable expressivity to small classical MLPs.
- Any utility comes from the **trainable photonic feature map**, not computational quantum advantage.
- The auxiliary-derivative trick is a **classical algorithmic choice** that improves stability.
- **No honest evidence of quantum advantage** was found under controlled conditions.
- The strongest predictor of accuracy was **classical model capacity** (parameter count), not quantum layer presence.
