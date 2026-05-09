"""Generate daniel_noise_study_phase1_phase2.ipynb"""
import json
import os

def md_cell(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code_cell(src: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

# =====================================================================
# Cell 1: Title
# =====================================================================
cells.append(md_cell(r"""# DV Photonic Noise Study for MerLin/Perceval QPINN
## Phase 1 & Phase 2 — ETH Quantum Hackathon 2026

This notebook studies physically correct **discrete-variable (DV) photonic noise** in the MerLin/Perceval quantum circuits used in the QPINN project.

**Important physics note:** The reference QPINN paper uses continuous-variable (CV) photonics (squeezed states, quadratures, homodyne detection). The implementation here uses **MerLin discrete-variable photonics** (Fock states, linear optical interferometers, photon-number detection). Therefore, CV noise models (finite squeezing, quadrature attenuation, displacement noise, Kerr-gate noise) are **not** the correct primary noise model. We study DV-native noise instead:

1. **Finite-shot sampling noise** — measurement statistics, not hardware decoherence.
2. **Photon loss / transmittance** — photons leak out of the intended Fock sector.
3. **Detector model** — PNR (photon-number resolving) vs threshold (click/no-click).
4. **Phase-shifter calibration noise** — random errors in interferometer phase settings.
5. **Source imperfections** — brightness, indistinguishability, multiphoton contamination ($g_2$).

The notebook first inspects the existing Phase 1 and Phase 2 quantum layers, then runs controlled noise sweeps, and reports quantitative metrics (RMSE, PDE residual, total loss, etc.).
"""))

# =====================================================================
# Cell 2: Imports and config
# =====================================================================
cells.append(code_cell(r"""import math, time, random, os, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# MerLin / Perceval
import merlin as ML
import perceval as pcvl

# Reproducibility
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

print(f"Device: {DEVICE}")
print(f"Torch:  {torch.__version__}")
print(f"MerLin: {getattr(ML, '__version__', 'unknown')}")

# ------------------------------------------------------------------
# FAST_MODE: quick but meaningful results for development
# ------------------------------------------------------------------
FAST_MODE = True

if FAST_MODE:
    print("\n=== FAST_MODE ===")
    PHASE1_EPOCHS = 100
    PHASE2_EPOCHS = 50
    SHOT_SEEDS = 3
else:
    print("\n=== FULL_MODE ===")
    PHASE1_EPOCHS = 300
    PHASE2_EPOCHS = 300
    SHOT_SEEDS = 5

# Noise sweep parameters
SHOTS_VALUES = [100, 250, 500, 1000, 2500, 5000, 10000]
TRANSMITTANCE_VALUES = [1.0, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70]
PHASE_ERROR_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
BRIGHTNESS_VALUES = [1.0, 0.98, 0.95, 0.90, 0.85]
INDISTINGUISHABILITY_VALUES = [1.0, 0.98, 0.95, 0.90, 0.80]
G2_VALUES = [0.0, 0.001, 0.005, 0.01, 0.02]

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
"""))

# =====================================================================
# Cell 3: Phase 1 problem, models, and training
# =====================================================================
cells.append(md_cell(r"""## Phase 1 — Problem, Models, and Training

Reuse the Phase 1 heat-equation setup and model definitions from `daniel_phase_1.ipynb`.
Domain: $x \in [0,1]$, $t \in [0,1]$, $\alpha = 0.1$.
"""))

# =====================================================================
# Cell 4: Phase 1 code
# =====================================================================
cells.append(code_cell(r"""# ------------------------------------------------------------
# Phase 1 Problem
# ------------------------------------------------------------
alpha_p1 = 0.1

def exact_u_p1(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-alpha_p1 * math.pi**2 * t) * torch.sin(math.pi * x)

def sample_interior_p1(n: int) -> torch.Tensor:
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    xt = torch.cat([x, t], dim=1)
    xt.requires_grad_(True)
    return xt

def sample_initial_p1(n: int):
    x = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.zeros_like(x)
    xt = torch.cat([x, t], dim=1)
    y = exact_u_p1(x, t)
    return xt, y

def sample_boundary_p1(n: int):
    n0 = n // 2
    n1 = n - n0
    t0 = torch.rand(n0, 1, device=DEVICE, dtype=DTYPE)
    t1 = torch.rand(n1, 1, device=DEVICE, dtype=DTYPE)
    x0 = torch.zeros_like(t0)
    x1 = torch.ones_like(t1)
    xt = torch.cat([torch.cat([x0, t0], dim=1), torch.cat([x1, t1], dim=1)], dim=0)
    y = torch.zeros(n, 1, device=DEVICE, dtype=DTYPE)
    return xt, y

def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True,
    )[0]

# ------------------------------------------------------------
# Phase 1 Models
# ------------------------------------------------------------
class ClassicalPINN_P1(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)

class ClassicalAuxPINN_P1(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.net(xt)
        return y[:, 0:1], y[:, 1:2]

class MerlinQPINN_P1(nn.Module):
    def __init__(self, feature_size: int = 4, quantum_output_size: int = 4, hidden: int = 16):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size),
        )
        self.quantum = ML.QuantumLayer.simple(
            input_size=feature_size, output_size=quantum_output_size,
        )
        self.readout = nn.Sequential(
            nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2),
        )
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        u = x * (1.0 - x) * q_u
        return u, ux_hat

def pde_residual_direct_p1(model, xt):
    u = model(xt)
    grad_u = gradients(u, xt)
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    u_xx = gradients(u_x, xt)[:, 0:1]
    return u_t - alpha_p1 * u_xx

def pde_residual_aux_p1(model, xt):
    u, ux_hat = model(xt)
    grad_u = gradients(u, xt)
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    ux_hat_x = gradients(ux_hat, xt)[:, 0:1]
    residual = u_t - alpha_p1 * ux_hat_x
    consistency = u_x - ux_hat
    return residual, consistency

def pde_residual_merlin_p1(model, xt):
    u, ux_hat = model(xt)
    grad_u = gradients(u, xt)
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    ux_hat_x = gradients(ux_hat, xt)[:, 0:1]
    residual = u_t - alpha_p1 * ux_hat_x
    consistency = u_x - ux_hat
    return residual, consistency

@dataclass
class TrainConfig_P1:
    epochs: int = 300
    n_f: int = 64
    n_i: int = 64
    n_b: int = 64
    lr: float = 1e-2
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 1.0
    lambda_consistency: float = 0.1
    print_every: int = 25

def train_model_p1(model, config: TrainConfig_P1, use_aux=False, is_merlin=False):
    model = model.to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    history = {"total": [], "pde": [], "ic": [], "bc": []}
    if is_merlin or use_aux:
        history["consistency"] = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        xt_f = sample_interior_p1(config.n_f)
        xt_i, y_i = sample_initial_p1(config.n_i)
        xt_b, y_b = sample_boundary_p1(config.n_b)
        if is_merlin or use_aux:
            if is_merlin:
                r_f, r_c = pde_residual_merlin_p1(model, xt_f)
            else:
                r_f, r_c = pde_residual_aux_p1(model, xt_f)
            loss_pde = mse(r_f, torch.zeros_like(r_f))
            loss_consistency = mse(r_c, torch.zeros_like(r_c))
            u_i, _ = model(xt_i)
            u_b, _ = model(xt_b)
        else:
            r_f = pde_residual_direct_p1(model, xt_f)
            loss_pde = mse(r_f, torch.zeros_like(r_f))
            loss_consistency = None
            u_i = model(xt_i)
            u_b = model(xt_b)
        loss_ic = mse(u_i, y_i)
        loss_bc = mse(u_b, y_b)
        loss = (config.lambda_pde * loss_pde + config.lambda_ic * loss_ic + config.lambda_bc * loss_bc)
        if loss_consistency is not None:
            loss = loss + config.lambda_consistency * loss_consistency
        loss.backward()
        optimizer.step()
        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        if loss_consistency is not None:
            history["consistency"].append(loss_consistency.item())
        if epoch % config.print_every == 0 or epoch == 1:
            msg = f"Epoch {epoch:4d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e} | ic={loss_ic.item():.3e} | bc={loss_bc.item():.3e}"
            if loss_consistency is not None:
                msg += f" | cons={loss_consistency.item():.3e}"
            print(msg)
    elapsed = time.time() - start
    return history, elapsed

def evaluate_model_p1(model, use_aux=False, is_merlin=False, nx=101, nt=101):
    with torch.no_grad():
        x = torch.linspace(0, 1, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
        t = torch.linspace(0, 1, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        xt = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        if use_aux or is_merlin:
            u_pred, _ = model(xt)
        else:
            u_pred = model(xt)
        u_pred = u_pred.reshape(nx, nt)
        u_true = exact_u_p1(X, T)
        diff = u_pred - u_true
        rel_l2 = torch.linalg.norm(diff) / torch.linalg.norm(u_true)
        rmse = torch.sqrt(torch.mean(diff**2))
        mae = torch.mean(torch.abs(diff))
        max_abs = torch.max(torch.abs(diff))
    # PDE residual on random interior points
    xi = torch.rand(2000, 1, device=DEVICE, dtype=DTYPE)
    ti = torch.rand(2000, 1, device=DEVICE, dtype=DTYPE)
    xti = torch.cat([xi, ti], dim=1)
    xti.requires_grad_(True)
    if is_merlin:
        r, _ = pde_residual_merlin_p1(model, xti)
    elif use_aux:
        r, _ = pde_residual_aux_p1(model, xti)
    else:
        r = pde_residual_direct_p1(model, xti)
    pde_mse = torch.mean(r**2).item()
    return {
        "rel_l2": rel_l2.item(), "rmse": rmse.item(), "mae": mae.item(),
        "max_err": max_abs.item(), "pde_mse": pde_mse,
        "X": X.cpu().numpy(), "T": T.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(), "u_true": u_true.cpu().numpy(),
    }

print("Phase 1 problem and models loaded.")
"""))

print("Written first 4 cells...")

# =====================================================================
# Cell 5: Phase 2 problem and models
# =====================================================================
cells.append(md_cell(r"""## Phase 2 — Problem, Models, and Training

Reuse the Phase 2 setup from `daniel_phase_2.ipynb`.
Domain: $x \in [-π/2, π/2]$, $t \in [0, 0.5]$, $\alpha = 0.30$.
Reference solution computed with finite differences + `scipy.integrate.solve_ivp`.
"""))

# =====================================================================
# Cell 6: Phase 2 code
# =====================================================================
cells.append(code_cell(r"""from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

ALPHA_P2 = 0.30
X_MIN_P2, X_MAX_P2 = -math.pi/2, math.pi/2
T_MIN_P2, T_MAX_P2 = 0.0, 0.5
SIGMA2_P2 = 0.2
X0_P2 = -math.pi/8

def initial_condition_p2(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.exp(-(x - X0_P2)**2 / (2 * SIGMA2_P2))

def solve_reference_p2(nx=401, nt_eval=401):
    x = np.linspace(X_MIN_P2, X_MAX_P2, nx)
    dx = x[1] - x[0]
    u0 = initial_condition_p2(x)
    u0_interior = u0[1:-1].astype(np.float64)
    n = nx - 2
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    L = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx**2)
    def rhs(t, u):
        return ALPHA_P2 * (L @ u)
    t_eval = np.linspace(T_MIN_P2, T_MAX_P2, nt_eval)
    sol = solve_ivp(rhs, (T_MIN_P2, T_MAX_P2), u0_interior, t_eval=t_eval, method='RK45')
    U = np.zeros((nx, nt_eval))
    U[1:-1, :] = sol.y
    U[0, :] = 0.0
    U[-1, :] = 0.0
    X_grid, T_grid = np.meshgrid(x, t_eval, indexing='ij')
    return X_grid, T_grid, U, x, t_eval

print("Computing Phase 2 reference solution...")
X_ref_p2, T_ref_p2, U_ref_p2, x_vec_p2, t_vec_p2 = solve_reference_p2(nx=401, nt_eval=401)
print(f"Reference grid: {X_ref_p2.shape}, range [{U_ref_p2.min():.4f}, {U_ref_p2.max():.4f}]")

def reference_at_p2(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().ravel()
    t_np = t.detach().cpu().numpy().ravel()
    interp = RegularGridInterpolator((x_vec_p2, t_vec_p2), U_ref_p2, bounds_error=False, fill_value=0.0)
    vals = interp(np.stack([x_np, t_np], axis=1))
    return torch.tensor(vals, dtype=DTYPE, device=DEVICE).view(-1, 1)

def bc_envelope_p2(x: torch.Tensor) -> torch.Tensor:
    a = math.pi / 2
    return (a**2 - x**2)

def bc_envelope_d1_p2(x: torch.Tensor) -> torch.Tensor:
    return -2.0 * x

class ClassicalDirectPINN_P2(nn.Module):
    def __init__(self, hidden=32, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, xt):
        q = self.net(xt)
        x = xt[:, 0:1]
        return bc_envelope_p2(x) * q

class ClassicalAuxPINN_P2(nn.Module):
    def __init__(self, hidden=32, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, xt):
        out = self.net(xt)
        q = out[:, 0:1]
        qx_hat = out[:, 1:2]
        x = xt[:, 0:1]
        env = bc_envelope_p2(x)
        env_d1 = bc_envelope_d1_p2(x)
        T = env * q
        Tx = env_d1 * q + env * qx_hat
        return T, Tx

class MerlinAuxQPINN_P2(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16, depth=1):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size),
        )
        if depth == 1:
            self.quantum = ML.QuantumLayer.simple(
                input_size=feature_size, output_size=quantum_output_size)
            self.post = nn.Identity()
        else:
            n_modes = feature_size + 1
            n_photons = math.ceil((feature_size + 1) / 2)
            builder = ML.CircuitBuilder(n_modes=n_modes)
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"L{i}")
            builder.add_angle_encoding(modes=list(range(feature_size)), name="input")
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"R{i}")
            circ = builder.to_pcvl_circuit()
            exp = pcvl.Experiment(m_circuit=circ)
            state_list = [0] * n_modes
            for k in range(n_photons):
                state_list[2*k] = 1
            exp.with_input(pcvl.BasicState(state_list))
            self.quantum = ML.QuantumLayer(
                experiment=exp, n_photons=n_photons,
                trainable_parameters=[f"L{i}" for i in range(depth)] + [f"R{i}" for i in range(depth)],
                input_parameters=["input"],
                measurement_strategy=ML.MeasurementStrategy.probs(),
            )
            q_out = self.quantum.output_size
            if q_out != quantum_output_size:
                self.post = ML.ModGrouping(input_size=q_out, groups=quantum_output_size)
            else:
                self.post = nn.Identity()
        self.depth = depth
        self.readout = nn.Sequential(
            nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 2),
        )
    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        if self.depth > 1:
            q = self.post(q)
        out = self.readout(q)
        q_u = out[:, 0:1]
        qx_hat = out[:, 1:2]
        env = bc_envelope_p2(x)
        env_d1 = bc_envelope_d1_p2(x)
        T = env * q_u
        Tx = env_d1 * q_u + env * qx_hat
        return T, Tx

class MerlinDirectQPINN_P2(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16, depth=1):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, feature_size),
        )
        if depth == 1:
            self.quantum = ML.QuantumLayer.simple(
                input_size=feature_size, output_size=quantum_output_size)
            self.post = nn.Identity()
        else:
            n_modes = feature_size + 1
            n_photons = math.ceil((feature_size + 1) / 2)
            builder = ML.CircuitBuilder(n_modes=n_modes)
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"L{i}")
            builder.add_angle_encoding(modes=list(range(feature_size)), name="input")
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"R{i}")
            circ = builder.to_pcvl_circuit()
            exp = pcvl.Experiment(m_circuit=circ)
            state_list = [0] * n_modes
            for k in range(n_photons):
                state_list[2*k] = 1
            exp.with_input(pcvl.BasicState(state_list))
            self.quantum = ML.QuantumLayer(
                experiment=exp, n_photons=n_photons,
                trainable_parameters=[f"L{i}" for i in range(depth)] + [f"R{i}" for i in range(depth)],
                input_parameters=["input"],
                measurement_strategy=ML.MeasurementStrategy.probs(),
            )
            q_out = self.quantum.output_size
            if q_out != quantum_output_size:
                self.post = ML.ModGrouping(input_size=q_out, groups=quantum_output_size)
            else:
                self.post = nn.Identity()
        self.depth = depth
        self.readout = nn.Sequential(
            nn.Linear(quantum_output_size, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        q = self.quantum(z)
        if self.depth > 1:
            q = self.post(q)
        q_u = self.readout(q)
        return bc_envelope_p2(x) * q_u

def sample_interior_p2(n):
    x = torch.empty(n, 1, device=DEVICE, dtype=DTYPE).uniform_(X_MIN_P2, X_MAX_P2)
    t = torch.empty(n, 1, device=DEVICE, dtype=DTYPE).uniform_(T_MIN_P2, T_MAX_P2)
    return torch.cat([x, t], dim=1)

def sample_initial_p2(n):
    x = torch.empty(n, 1, device=DEVICE, dtype=DTYPE).uniform_(X_MIN_P2, X_MAX_P2)
    t = torch.full((n, 1), T_MIN_P2, device=DEVICE, dtype=DTYPE)
    return torch.cat([x, t], dim=1), reference_at_p2(x, torch.zeros_like(x))

def sample_boundary_p2(n):
    side = torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    x = torch.where(side < 0.5, torch.full_like(side, X_MIN_P2), torch.full_like(side, X_MAX_P2))
    t = torch.empty(n, 1, device=DEVICE, dtype=DTYPE).uniform_(T_MIN_P2, T_MAX_P2)
    return torch.cat([x, t], dim=1), torch.zeros(n, 1, device=DEVICE, dtype=DTYPE)

def pde_residual_direct_p2(model, xt):
    xt = xt.requires_grad_(True)
    T = model(xt)
    grad_T = gradients(T, xt)
    T_t = grad_T[:, 1:2]
    T_x = grad_T[:, 0:1]
    T_xx = gradients(T_x, xt)[:, 0:1]
    return T_t - ALPHA_P2 * T_xx

def pde_residual_aux_p2(model, xt):
    xt = xt.requires_grad_(True)
    T, Tx = model(xt)
    T_t = gradients(T, xt)[:, 1:2]
    Tx_x = gradients(Tx, xt)[:, 0:1]
    r_f = T_t - ALPHA_P2 * Tx_x
    T_x_auto = gradients(T, xt)[:, 0:1]
    r_c = T_x_auto - Tx
    return r_f, r_c

@dataclass
class TrainConfig_P2:
    epochs: int = 300
    n_f: int = 128
    n_i: int = 128
    n_b: int = 128
    lr: float = 1e-2
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 1.0
    lambda_consistency: float = 0.1
    print_every: int = 25

def train_model_p2(model, config: TrainConfig_P2, mode="aux"):
    model = model.to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    history = {"total": [], "pde": [], "ic": [], "bc": [], "consistency": []}
    start = time.time()
    is_aux = "aux" in mode
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        xt_f = sample_interior_p2(config.n_f)
        xt_i, y_i = sample_initial_p2(config.n_i)
        xt_b, y_b = sample_boundary_p2(config.n_b)
        if is_aux:
            r_f, r_c = pde_residual_aux_p2(model, xt_f)
            loss_pde = mse(r_f, torch.zeros_like(r_f))
            loss_cons = mse(r_c, torch.zeros_like(r_c))
            T_i, _ = model(xt_i)
            T_b, _ = model(xt_b)
        else:
            r_f = pde_residual_direct_p2(model, xt_f)
            loss_pde = mse(r_f, torch.zeros_like(r_f))
            loss_cons = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
            T_i = model(xt_i)
            T_b = model(xt_b)
        loss_ic = mse(T_i, y_i)
        loss_bc = mse(T_b, y_b)
        loss = (config.lambda_pde * loss_pde + config.lambda_ic * loss_ic
                + config.lambda_bc * loss_bc + config.lambda_consistency * loss_cons)
        loss.backward()
        optimizer.step()
        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["consistency"].append(loss_cons.item())
        if config.print_every > 0 and (epoch % config.print_every == 0 or epoch == 1):
            print(f"Epoch {epoch:4d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"ic={loss_ic.item():.3e} | bc={loss_bc.item():.3e} | cons={loss_cons.item():.3e}")
    elapsed = time.time() - start
    return history, elapsed

def evaluate_model_p2(model, mode="aux", nx=51, nt=51):
    model.eval()
    with torch.no_grad():
        x = torch.linspace(X_MIN_P2, X_MAX_P2, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
        t = torch.linspace(T_MIN_P2, T_MAX_P2, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
        Xg, Tg = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        xt = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
        if "aux" in mode:
            T_pred, _ = model(xt)
        else:
            T_pred = model(xt)
        T_pred = T_pred.reshape(nx, nt).cpu().numpy()
    interp = RegularGridInterpolator((x_vec_p2, t_vec_p2), U_ref_p2, bounds_error=False, fill_value=0.0)
    X_np = Xg.cpu().numpy()
    T_np = Tg.cpu().numpy()
    T_ref_grid = interp(np.stack([X_np.ravel(), T_np.ravel()], axis=1)).reshape(nx, nt)
    diff = T_pred - T_ref_grid
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(T_ref_grid)
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    nmse = np.mean(diff**2) / np.mean(T_ref_grid**2)
    # PDE residual
    xi = torch.rand(min(2000, nx*nt), 1, device=DEVICE, dtype=DTYPE).uniform_(X_MIN_P2, X_MAX_P2)
    ti = torch.rand(min(2000, nx*nt), 1, device=DEVICE, dtype=DTYPE).uniform_(T_MIN_P2, T_MAX_P2)
    xti = torch.cat([xi, ti], dim=1)
    xti.requires_grad_(True)
    if "aux" in mode:
        r_f, _ = pde_residual_aux_p2(model, xti)
    else:
        r_f = pde_residual_direct_p2(model, xti)
    pde_mse = torch.mean(r_f**2).item()
    # IC and BC
    xt_ic, y_ic = sample_initial_p2(500)
    T_ic_pred = model(xt_ic)[0] if "aux" in mode else model(xt_ic)
    ic_mse = torch.mean((T_ic_pred - y_ic)**2).item()
    xt_bc, y_bc = sample_boundary_p2(500)
    T_bc_pred = model(xt_bc)[0] if "aux" in mode else model(xt_bc)
    bc_mse = torch.mean((T_bc_pred - y_bc)**2).item()
    return {
        "rel_l2": rel_l2, "rmse": rmse, "mae": mae, "max_err": max_err, "nmse": nmse,
        "pde_mse": pde_mse, "ic_mse": ic_mse, "bc_mse": bc_mse,
        "T_pred": T_pred, "T_ref": T_ref_grid, "X": X_np, "T": T_np,
    }

print("Phase 2 problem and models loaded.")
"""))

print("Written Phase 2 cells...")
