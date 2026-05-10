"""HOM-aware noise study for the trained MerLin QPINN.

Why this script exists
----------------------
MerLin 0.2.3's QuantumLayer pipeline only consumes `brightness` and
`transmittance` from a `pcvl.NoiseModel` (see merlin/measurement/photon_loss.py
lines 373-374). The four HOM-physics channels --
`indistinguishability`, `g2`, `phase_error`, `phase_imprecision` -- are
silently dropped, which is why those rows in the original notebook noise
sweep are perfectly flat.

To actually expose the trained QPINN to HOM physics we bypass MerLin at
*evaluation time* and route the same circuit, the same trained weights, and
the same per-sample input parameters through Perceval's Processor with the
full NoiseModel attached. We then re-aggregate the resulting Fock-state
distribution onto the same 4-d output basis MerLin uses (via ML.ModGrouping)
and forward through the model's classical readout. Training is unchanged --
we keep MerLin for autograd.

This is the "trained ideal, evaluated noisy" robustness study.

Run with:
  ETH_env/bin/python hom_aware_noise_study.py
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import copy
import math
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed

import merlin as ML
import perceval as pcvl
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

DEVICE = torch.device('cpu')
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# -----------------------------------------------------------------------------
# Replicate the exact model classes used by the notebook (so we can load the
# saved FOCK checkpoints unchanged).
# -----------------------------------------------------------------------------

class _QPINNQuantum(nn.Module):
    def __init__(self, core, post):
        super().__init__()
        self.quantum_layer = core
        self.post_processing = post
        self.output_size = (post.output_size if hasattr(post, 'output_size')
                            else int(core.output_size))
        self.circuit = getattr(core, 'circuit', None)

    def forward(self, x):
        return self.post_processing(self.quantum_layer(x))


def _build_qpinn_quantum(feature_size, quantum_output_size, use_fock=True):
    base = ML.QuantumLayer.simple(input_size=feature_size, output_size=quantum_output_size)
    if not use_fock:
        return base, dict(origin='QuantumLayer.simple', computation_space='UNBUNCHED')
    bc = base.quantum_layer
    fock_core = ML.QuantumLayer(
        experiment=bc.experiment,
        n_photons=bc.n_photons,
        trainable_parameters=bc.trainable_parameters,
        input_parameters=bc.input_parameters,
        measurement_strategy=ML.MeasurementStrategy.probs(
            computation_space=ML.ComputationSpace.FOCK),
    )
    fock_core.load_state_dict(bc.state_dict(), strict=False)
    if int(fock_core.output_size) != int(quantum_output_size):
        post = ML.ModGrouping(input_size=int(fock_core.output_size),
                              output_size=int(quantum_output_size))
    else:
        post = nn.Identity()
    info = dict(origin='QuantumLayer.simple -> FOCK rewrap', computation_space='FOCK',
                fock_output_size=int(fock_core.output_size))
    return _QPINNQuantum(fock_core, post), info


class MerlinQPINNPhase1(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.feature_map = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size))
        self.quantum, self.quantum_build_info = _build_qpinn_quantum(feature_size, quantum_output_size, use_fock=True)
        self.readout = nn.Sequential(nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)
        u = x * (1.0 - x) * out[:, 0:1]
        return u, out[:, 1:2]


# Phase 2 reference solution (heat equation on [-pi/2, pi/2])
ALPHA2 = 0.30
XMIN2, XMAX2 = -math.pi / 2, math.pi / 2
TMIN2, TMAX2 = 0.0, 0.5
SIGMA2 = 0.2
X0 = -math.pi / 8

def initial_condition2(x):
    return 0.5 * np.exp(-(x - X0) ** 2 / (2 * SIGMA2))

def solve_reference2(nx=301, nt_eval=301):
    x = np.linspace(XMIN2, XMAX2, nx)
    dx = x[1] - x[0]
    u0 = initial_condition2(x)
    u0_int = u0[1:-1].astype(np.float64)
    n = nx - 2
    L = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / (dx ** 2)
    sol = solve_ivp(lambda t, u: ALPHA2 * (L @ u), (TMIN2, TMAX2), u0_int,
                    t_eval=np.linspace(TMIN2, TMAX2, nt_eval), method='RK45')
    U = np.zeros((nx, nt_eval))
    U[1:-1, :] = sol.y
    return x, np.linspace(TMIN2, TMAX2, nt_eval), U

print('Building reference solution...')
x_ref2, t_ref2, U_ref2 = solve_reference2(nx=301, nt_eval=301)
interp_ref2 = RegularGridInterpolator((x_ref2, t_ref2), U_ref2, bounds_error=False, fill_value=0.0)


def bc_env2(x):
    a = math.pi / 2
    return a * a - x * x

def bc_env2_d1(x):
    return -2.0 * x


class MerlinAuxQPINNPhase2(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16, depth=1):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.feature_map = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size))
        self.quantum, self.quantum_build_info = _build_qpinn_quantum(feature_size, quantum_output_size, use_fock=True)
        self.post = nn.Identity()
        self.readout = nn.Sequential(nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)
        env = bc_env2(x)
        env_d1 = bc_env2_d1(x)
        T = env * out[:, 0:1]
        Tx = env_d1 * out[:, 0:1] + env * out[:, 1:2]
        return T, Tx


# -----------------------------------------------------------------------------
# Perceval-based replacement for the MerLin quantum layer (forward-only,
# non-differentiable, but supports the *full* NoiseModel including HOM channels).
# -----------------------------------------------------------------------------

# Module-level worker (no `self` pickled). Takes only pickle-safe inputs:
# the trained LI/RI vectors, the input vector, an input-state TUPLE (we
# reconstruct pcvl.BasicState inside the worker), and the noise kwargs dict.
def _pcvl_row(li_arr, ri_arr, x_row, input_state_tuple, noise_kwargs, fock_keys_tuples):
    import perceval as _pcvl
    import merlin as _ML
    import copy as _copy
    import numpy as _np
    # Re-derive a fresh symbolic circuit each call. Cheaper than passing the
    # circuit through joblib (it has C-level state internally).
    seed = _ML.QuantumLayer.simple(input_size=4, output_size=4)
    circ = _copy.deepcopy(seed.quantum_layer.circuit)
    params = circ.get_parameters()

    li_names = [n for j in range(10) for n in (f'LI_simple_li{j}', f'LI_simple_lo{j}')]
    ri_names = [n for j in range(10) for n in (f'RI_simple_li{j}', f'RI_simple_lo{j}')]
    bind = {}
    for n, v in zip(li_names, li_arr):
        bind[n] = float(v)
    for n, v in zip(ri_names, ri_arr):
        bind[n] = float(v)
    for k in range(4):
        bind[f'input{k+1}'] = float(x_row[k])
    for p in params:
        if p.name in bind:
            p.set_value(bind[p.name])

    exp = _pcvl.Experiment(m_circuit=circ)
    exp.with_input(_pcvl.BasicState(list(input_state_tuple)))
    if noise_kwargs:
        exp.noise = _pcvl.NoiseModel(**noise_kwargs)
    proc = _pcvl.Processor('SLOS', exp)
    proc.min_detected_photons_filter(0)
    res = _pcvl.algorithm.Sampler(proc).probs()['results']

    out = _np.zeros(len(fock_keys_tuples), dtype=_np.float64)
    key_to_idx = {k: i for i, k in enumerate(fock_keys_tuples)}
    for k, p in res.items():
        tup = tuple(int(v) for v in k)
        idx = key_to_idx.get(tup)
        if idx is not None:
            out[idx] = p
    return out


class PercevalNoisyQuantum(nn.Module):
    """Drop-in replacement for the QPINN's quantum block that runs each
    forward sample through `pcvl.Processor` with the *full* `pcvl.NoiseModel`.

    Forward is non-differentiable; evaluation only.
    """
    def __init__(self, fock_keys, input_state_tuple, li_arr, ri_arr,
                 post_processing, noise_kwargs=None, n_jobs=1):
        super().__init__()
        self._fock_keys_tuples = [tuple(int(v) for v in k) for k in fock_keys]
        self._input_state_tuple = tuple(int(v) for v in input_state_tuple)
        self._li_arr = np.asarray(li_arr, dtype=np.float64)
        self._ri_arr = np.asarray(ri_arr, dtype=np.float64)
        self.post_processing = post_processing
        self.output_size = (post_processing.output_size if hasattr(post_processing, 'output_size')
                            else len(self._fock_keys_tuples))
        self._noise_kwargs = dict(noise_kwargs or {})
        self._n_jobs = n_jobs

    def forward(self, x):
        x_np = x.detach().cpu().numpy().astype(float)
        N = x_np.shape[0]
        if self._n_jobs and self._n_jobs > 1 and N > 8:
            rows = Parallel(n_jobs=self._n_jobs, backend='loky', verbose=0)(
                delayed(_pcvl_row)(
                    self._li_arr, self._ri_arr, x_np[i],
                    self._input_state_tuple, self._noise_kwargs, self._fock_keys_tuples
                ) for i in range(N))
            probs = np.stack(rows, axis=0)
        else:
            probs = np.stack([_pcvl_row(
                self._li_arr, self._ri_arr, x_np[i],
                self._input_state_tuple, self._noise_kwargs, self._fock_keys_tuples
            ) for i in range(N)], axis=0)
        probs_t = torch.from_numpy(probs).to(dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            return self.post_processing(probs_t)


# -----------------------------------------------------------------------------
# Build a HOM-noise-aware version of a trained model: replace its quantum
# block with a PercevalNoisyQuantum that has the same trained params bound.
# -----------------------------------------------------------------------------

def make_noisy_model_for_eval(model, noise_kwargs, n_jobs=1):
    m_noisy = copy.deepcopy(model)
    quantum = m_noisy.quantum  # _QPINNQuantum
    core = quantum.quantum_layer
    post = quantum.post_processing
    fock_keys = list(core.output_keys)
    # Convert the input state to a tuple so it can be pickled to workers.
    input_state_tuple = tuple(int(v) for v in core.input_state)
    sd = core.state_dict()
    perc = PercevalNoisyQuantum(
        fock_keys=fock_keys,
        input_state_tuple=input_state_tuple,
        li_arr=sd['LI_simple'].detach().cpu().numpy(),
        ri_arr=sd['RI_simple'].detach().cpu().numpy(),
        post_processing=post,
        noise_kwargs=noise_kwargs,
        n_jobs=n_jobs,
    )
    m_noisy.quantum = perc
    m_noisy.eval()
    return m_noisy


# -----------------------------------------------------------------------------
# Evaluation: only RMSE / IC / BC (no autograd through Perceval).
# -----------------------------------------------------------------------------

def alpha1_exact_u1(x, t):
    return torch.exp(-0.1 * math.pi**2 * t) * torch.sin(math.pi * x)


def evaluate_rmse(model, phase, nx=41, nt=41, n_jobs=1):
    model.eval()
    if phase == 'phase1':
        x = torch.linspace(0, 1, nx, dtype=DTYPE).view(-1, 1)
        t = torch.linspace(0, 1, nt, dtype=DTYPE).view(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        u_pred, _ = model(xt)
        u_true = alpha1_exact_u1(X, T).reshape(-1, 1)
    else:
        x = torch.linspace(XMIN2, XMAX2, nx, dtype=DTYPE).view(-1, 1)
        t = torch.linspace(TMIN2, TMAX2, nt, dtype=DTYPE).view(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        u_pred, _ = model(xt)
        pts = np.stack([X.cpu().numpy().ravel(), T.cpu().numpy().ravel()], axis=1)
        u_true = torch.tensor(interp_ref2(pts), dtype=DTYPE).view(-1, 1)
    diff = (u_pred - u_true).detach()
    rmse = float(torch.sqrt(torch.mean(diff ** 2)))
    mae = float(torch.mean(torch.abs(diff)))
    return rmse, mae


# -----------------------------------------------------------------------------
# Run sweeps
# -----------------------------------------------------------------------------

CKPTS = {
    'phase1': 'results/phase1_merlin_for_noise_fock.pt',
    'phase2': 'results/phase2_merlin_for_noise_fock.pt',
}

def load_model(phase):
    if phase == 'phase1':
        m = MerlinQPINNPhase1()
    else:
        m = MerlinAuxQPINNPhase2()
    sd = torch.load(CKPTS[phase], map_location='cpu')
    m.load_state_dict(sd)
    return m


def run_sweep(model, phase, sweep, family, eval_grid=(41, 41), n_jobs_eval=64):
    rows = []
    for label, kwargs in sweep:
        t0 = time.time()
        m_noisy = make_noisy_model_for_eval(model, kwargs, n_jobs=n_jobs_eval)
        rmse, mae = evaluate_rmse(m_noisy, phase, nx=eval_grid[0], nt=eval_grid[1], n_jobs=n_jobs_eval)
        rows.append({
            'phase': phase, 'family': family, 'label': label,
            'rmse': rmse, 'mae': mae, 'eval_seconds': time.time() - t0,
            **{f'noise_{k}': v for k, v in kwargs.items()},
        })
        print(f'  {phase:6s} {family:22s} {label:30s}  RMSE={rmse:.4f}  ({rows[-1]["eval_seconds"]:.1f}s)')
    return rows


def main():
    print('Loading trained models (FOCK checkpoints)...')
    m1 = load_model('phase1')
    m2 = load_model('phase2')

    sweeps = {
        'indistinguishability': [(f'indist={v:.2f}', {'indistinguishability': v}) for v in [1.0, 0.98, 0.95, 0.90, 0.80, 0.50, 0.0]],
        'g2':                   [(f'g2={v:.3f}',     {'g2': v})                   for v in [0.0, 0.001, 0.005, 0.01, 0.05, 0.10]],
        'phase_error':          [(f'pe={v:.3f}',     {'phase_error': v})          for v in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.30]],
        'phase_imprecision':    [(f'pi={v:.3f}',     {'phase_imprecision': v})    for v in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.30]],
        'transmittance_ref':    [(f'T={v:.2f}',      {'transmittance': v})        for v in [1.0, 0.95, 0.90, 0.80]],
    }

    all_rows = []
    eval_grid = (41, 41)  # 1681 samples per setting; per-sample Perceval call ~ms
    n_jobs = 64

    for family, items in sweeps.items():
        print(f'\n=== {family} ===')
        all_rows += run_sweep(m1, 'phase1', items, family, eval_grid=eval_grid, n_jobs_eval=n_jobs)
        all_rows += run_sweep(m2, 'phase2', items, family, eval_grid=eval_grid, n_jobs_eval=n_jobs)

    df = pd.DataFrame(all_rows)
    out_csv = 'results/hom_aware_noise_sweep.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved -> {out_csv}   ({len(df)} rows)')

    # Compact summary
    print('\n=== Summary (RMSE) ===')
    for family in sweeps:
        sub = df[df['family'] == family]
        for phase in ['phase1', 'phase2']:
            s = sub[sub['phase'] == phase][['label', 'rmse']]
            print(f'\n{phase}  /  {family}')
            print(s.to_string(index=False))


if __name__ == '__main__':
    main()
