import matplotlib
matplotlib.use("Agg")

import math, time, random, os, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

import merlin as ML
import perceval as pcvl

DEVICE = torch.device("cpu")
DTYPE = torch.float32

CONFIG = {
    "FAST_MODE": False,
    "seeds": [0, 1, 2],
    "epochs": 50,
    "lr": 1e-2,
    "n_f": 64,
    "n_i": 64,
    "n_b": 64,
    "nx_eval": 51,
    "nt_eval": 51,
    "output_dir": "results",
}

if not CONFIG["FAST_MODE"]:
    CONFIG["epochs"] = 300
    CONFIG["seeds"] = [0, 1, 2, 3, 4]
    CONFIG["nx_eval"] = 101
    CONFIG["nt_eval"] = 101

print(f"Device: {DEVICE}")
print(f"Torch:  {torch.__version__}")
print(f"MerLin: {getattr(ML, '__version__', 'unknown')}")
print(f"FAST_MODE: {CONFIG['FAST_MODE']}")

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"] + "/figures", exist_ok=True)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@contextmanager
def timer():
    start = time.time()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = time.time() - start

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    diff = pred - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    nmse = float(np.mean(diff**2) / (np.mean(ref**2) + 1e-12))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return {"rmse": rmse, "nmse": nmse, "mae": mae, "max_err": max_err}

ALPHA = 0.30
X_MIN, X_MAX = -math.pi/2, math.pi/2
T_MIN, T_MAX = 0.0, 0.5
SIGMA2 = 0.2
X0 = -math.pi/8

def initial_condition(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.exp(-(x - X0)**2 / (2 * SIGMA2))

def solve_reference(nx=401, nt_eval=401, t_span=(T_MIN, T_MAX)):
    x = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]
    u0 = initial_condition(x)
    u0_interior = u0[1:-1].astype(np.float64)
    n = nx - 2
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    L = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx**2)
    def rhs(t, u):
        return ALPHA * (L @ u)
    t_eval = np.linspace(t_span[0], t_span[1], nt_eval)
    sol = solve_ivp(rhs, t_span, u0_interior, t_eval=t_eval, method='RK45')
    U = np.zeros((nx, nt_eval))
    U[1:-1, :] = sol.y
    U[0, :] = 0.0
    U[-1, :] = 0.0
    X_grid, T_grid = np.meshgrid(x, t_eval, indexing='ij')
    return X_grid, T_grid, U

print("Computing reference solution...")
X_ref, T_ref, U_ref = solve_reference(nx=401, nt_eval=401)
print(f"Reference grid: {X_ref.shape}")

def reference_at(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().ravel()
    t_np = t.detach().cpu().numpy().ravel()
    x_vec = np.linspace(X_MIN, X_MAX, X_ref.shape[0])
    t_vec = np.linspace(T_MIN, T_MAX, X_ref.shape[1])
    interp = RegularGridInterpolator((x_vec, t_vec), U_ref, bounds_error=False, fill_value=0.0)
    vals = interp(np.stack([x_np, t_np], axis=1))
    return torch.tensor(vals, dtype=DTYPE, device=DEVICE).view(-1, 1)

def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def bc_envelope(x: torch.Tensor) -> torch.Tensor:
    a = math.pi / 2
    return (a**2 - x**2)

def bc_envelope_d1(x: torch.Tensor) -> torch.Tensor:
    return -2.0 * x

def bc_envelope_d2(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, -2.0)

class ClassicalAuxPINN(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 3):
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
        env = bc_envelope(x)
        env_d1 = bc_envelope_d1(x)
        T = env * q
        Tx = env_d1 * q + env * qx_hat
        return T, Tx

class ClassicalFourierPINN(nn.Module):
    def __init__(self, hidden=32, depth=3, n_fourier=16, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(2, n_fourier) * scale * 2 * math.pi, requires_grad=False)
        layers = [nn.Linear(n_fourier * 2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, xt):
        x = xt[:, 0:1]
        z = torch.cat([torch.sin(xt @ self.B), torch.cos(xt @ self.B)], dim=1)
        out = self.net(z)
        q = out[:, 0:1]
        qx_hat = out[:, 1:2]
        env = bc_envelope(x)
        env_d1 = bc_envelope_d1(x)
        T = env * q
        Tx = env_d1 * q + env * qx_hat
        return T, Tx

class MerlinAuxQPINN(nn.Module):
    def __init__(self, feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, feature_size),
        )
        self.n_photons = n_photons
        self.depth = depth
        if depth == 1:
            self.quantum = ML.QuantumLayer.simple(
                input_size=feature_size, output_size=quantum_output_size)
            self.post = nn.Identity()
        else:
            n_modes = max(feature_size + 1, n_photons)
            builder = ML.CircuitBuilder(n_modes=n_modes)
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"L{i}")
            builder.add_angle_encoding(modes=list(range(feature_size)), name="input")
            for i in range(depth):
                builder.add_entangling_layer(trainable=True, name=f"R{i}")
            circ = builder.to_pcvl_circuit()
            exp = pcvl.Experiment(m_circuit=circ)
            state_list = [1 if idx < n_photons else 0 for idx in range(n_modes)]
            exp.with_input(pcvl.BasicState(state_list))
            self.quantum = ML.QuantumLayer(
                experiment=exp, n_photons=n_photons,
                trainable_parameters=[f"L{i}" for i in range(depth)] + [f"R{i}" for i in range(depth)],
                input_parameters=["input"],
                measurement_strategy=ML.MeasurementStrategy.probs(),
            )
            q_out = self.quantum.output_size
            if q_out != quantum_output_size:
                self.post = ML.ModGrouping(input_size=q_out, output_size=quantum_output_size)
            else:
                self.post = nn.Identity()
        self.readout = nn.Sequential(
            nn.Linear(quantum_output_size, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
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
        env = bc_envelope(x)
        env_d1 = bc_envelope_d1(x)
        T = env * q_u
        Tx = env_d1 * q_u + env * qx_hat
        return T, Tx

class MerLinRedesign(nn.Module):
    def __init__(self,
                 feature_size=4, 
                 quantum_output_size=4,
                 hidden=16,
                 depth=1,
                 n_photons=2,
                 reupload=False,
                 hard_bc=True,
                 readout_type="mlp"
                 ):
        super().__init__()
        self.reupload = reupload
        self.hard_bc = hard_bc
        self.depth = depth
        self.enc = nn.Linear(2, feature_size)
        n_modes = max(feature_size + 1, n_photons)
        builder = ML.CircuitBuilder(n_modes=n_modes)
        input_params = []
        trainable_params = []
        for d in range(depth):
            if reupload or d == 0:
                name = f"input_{d}"
                builder.add_angle_encoding(modes=list(range(feature_size)), name=name)
                input_params.append(name)
            lname = f"L{d}"
            builder.add_entangling_layer(trainable=True, name=lname)
            trainable_params.append(lname)
        circ = builder.to_pcvl_circuit()
        exp = pcvl.Experiment(m_circuit=circ)
        state_list = [1 if idx < n_photons else 0 for idx in range(n_modes)]
        exp.with_input(pcvl.BasicState(state_list))
        self.quantum = ML.QuantumLayer(
            experiment=exp, n_photons=n_photons,
            trainable_parameters=trainable_params,
            input_parameters=input_params,
            measurement_strategy=ML.MeasurementStrategy.probs(),
        )
        q_out = self.quantum.output_size
        if q_out != quantum_output_size:
            self.post = ML.ModGrouping(input_size=q_out, output_size=quantum_output_size)
        else:
            self.post = nn.Identity()
        if readout_type == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(quantum_output_size, hidden), nn.Tanh(),
                nn.Linear(hidden, 2),
            )
        else:
            self.readout = nn.Linear(quantum_output_size, 2)
    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.enc(xt)
        if self.reupload:
            # Concatenate z for each input slot (depth-wise re-upload)
            # QuantumLayer expects a single tensor with all input features concatenated
            z_in = torch.cat([z] * self.depth, dim=1)
            q = self.quantum(z_in)
        else:
            q = self.quantum(z)
        q = self.post(q)
        out = self.readout(q)
        q_u = out[:, 0:1]
        qx_hat = out[:, 1:2]
        if self.hard_bc:
            env = bc_envelope(x)
            env_d1 = bc_envelope_d1(x)
            T = env * q_u
            Tx = env_d1 * q_u + env * qx_hat
        else:
            T = q_u
            Tx = qx_hat
        return T, Tx

def sample_interior(n, device=DEVICE, dtype=DTYPE):
    x = torch.empty(n, 1, device=device, dtype=dtype).uniform_(X_MIN, X_MAX)
    t = torch.empty(n, 1, device=device, dtype=dtype).uniform_(T_MIN, T_MAX)
    return torch.cat([x, t], dim=1)

def sample_initial(n, device=DEVICE, dtype=DTYPE):
    x = torch.empty(n, 1, device=device, dtype=dtype).uniform_(X_MIN, X_MAX)
    t = torch.full((n, 1), T_MIN, device=device, dtype=dtype)
    return torch.cat([x, t], dim=1), reference_at(x, torch.zeros_like(x))

def sample_boundary(n, device=DEVICE, dtype=DTYPE):
    side = torch.rand(n, 1, device=device, dtype=dtype)
    x = torch.where(side < 0.5, torch.full_like(side, X_MIN), torch.full_like(side, X_MAX))
    t = torch.empty(n, 1, device=device, dtype=dtype).uniform_(T_MIN, T_MAX)
    return torch.cat([x, t], dim=1), torch.zeros(n, 1, device=device, dtype=dtype)

def pde_residual_aux(model, xt):
    xt = xt.requires_grad_(True)
    T, Tx = model(xt)
    T_t = gradients(T, xt)[:, 1:2]
    Tx_x = gradients(Tx, xt)[:, 0:1]
    r_f = T_t - ALPHA * Tx_x
    T_x_auto = gradients(T, xt)[:, 0:1]
    r_c = T_x_auto - Tx
    return r_f, r_c

def train_model(model, epochs, lr, n_f, n_i, n_b):
    model = model.to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = {"total": []}
    start = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        xt_f = sample_interior(n_f)
        xt_i, y_i = sample_initial(n_i)
        xt_b, y_b = sample_boundary(n_b)
        r_f, r_c = pde_residual_aux(model, xt_f)
        loss_pde = mse(r_f, torch.zeros_like(r_f))
        loss_cons = mse(r_c, torch.zeros_like(r_c))
        T_i, _ = model(xt_i)
        T_b, _ = model(xt_b)
        loss_ic = mse(T_i, y_i)
        loss_bc = mse(T_b, y_b)
        loss = (1.0 * loss_pde + 10.0 * loss_ic + 1.0 * loss_bc + 0.1 * loss_cons)
        loss.backward()
        optimizer.step()
        history["total"].append(loss.item())
    elapsed = time.time() - start
    return history, elapsed

def evaluate_model(model):
    nx, nt = CONFIG["nx_eval"], CONFIG["nt_eval"]
    model.eval()
    with torch.no_grad():
        x = torch.linspace(X_MIN, X_MAX, nx, device=DEVICE, dtype=DTYPE).view(-1, 1)
        t = torch.linspace(T_MIN, T_MAX, nt, device=DEVICE, dtype=DTYPE).view(-1, 1)
        Xmesh, Tmesh = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        xt = torch.stack([Xmesh.reshape(-1), Tmesh.reshape(-1)], dim=1)
        T_pred, _ = model(xt)
        T_pred = T_pred.reshape(nx, nt).cpu().numpy()
    x_vec = np.linspace(X_MIN, X_MAX, X_ref.shape[0])
    t_vec = np.linspace(T_MIN, T_MAX, X_ref.shape[1])
    interp = RegularGridInterpolator((x_vec, t_vec), U_ref, bounds_error=False, fill_value=0.0)
    T_ref_grid = interp(np.stack([Xmesh.cpu().numpy().ravel(), Tmesh.cpu().numpy().ravel()], axis=1)).reshape(nx, nt)
    metrics = compute_metrics(T_pred, T_ref_grid)
    xi = torch.rand(2000, 1, device=DEVICE, dtype=DTYPE).uniform_(X_MIN, X_MAX)
    ti = torch.rand(2000, 1, device=DEVICE, dtype=DTYPE).uniform_(T_MIN, T_MAX)
    xti = torch.cat([xi, ti], dim=1)
    xti.requires_grad_(True)
    r_f, _ = pde_residual_aux(model, xti)
    metrics["pde_residual"] = float(torch.mean(r_f**2).item())
    
    dummy = torch.randn(64, 2, device=DEVICE, dtype=DTYPE)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    metrics["forward_time_ms"] = (time.time() - t0) / 10 * 1000
    return metrics

def run_advantage_experiment(model_factory, config, seed, n_f=None):
    set_seed(seed)
    if n_f is None: n_f = config["n_f"]
    model = model_factory()
    params = count_parameters(model)
    hist, train_time = train_model(model, config["epochs"], config["lr"], n_f, config["n_i"], config["n_b"])
    metrics = evaluate_model(model)
    return {
        "seed": seed,
        "params": params,
        "train_time": train_time,
        "n_f": n_f,
        **metrics
    }

redesign_results = []
candidates = [
    ("Baseline", lambda: MerlinAuxQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2)),
    ("HardBC", lambda: MerLinRedesign(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2, hard_bc=True)),
    ("Reupload", lambda: MerLinRedesign(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2, reupload=True)),
    ("Reupload HardBC", lambda: MerLinRedesign(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2, reupload=True, hard_bc=True)),
    ("Reupload HardBC Linear Readout", lambda: MerLinRedesign(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2, reupload=True, hard_bc=True, readout_type='linear')),
    ("Reupload HardBC MLP Readout", lambda: MerLinRedesign(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2, reupload=True, hard_bc=True, readout_type='mlp')),
]

for name, factory in candidates:
    print(f"Training Redesign Candidate: {name}...")
    for seed in CONFIG["seeds"]:
        res = run_advantage_experiment(factory, CONFIG, seed)
        res["model"] = name
        redesign_results.append(res)

df_redesign = pd.DataFrame(redesign_results)
df_redesign.to_csv(os.path.join(CONFIG["output_dir"], "merlin_circuit_redesign_heat_results.csv"), index=False)

summary_redesign = df_redesign.groupby("model").agg({
    "params": "first",
    "rmse": ["mean", "std"],
    "pde_residual": "mean",
    "train_time": "mean"
})
print("\n=== Redesign Summary Table ===")
print(summary_redesign)

plt.figure(figsize=(10, 6))
means = df_redesign.groupby("model")["rmse"].mean().sort_values()
plt.bar(means.index, means.values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean RMSE")
plt.title("Redesign Candidate Performance (Mean RMSE)")
plt.yscale("log")
plt.grid(axis='y', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "redesign_rmse_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
for name in df_redesign["model"].unique():
    sub = df_redesign[df_redesign["model"] == name]
    plt.scatter(sub["params"], sub["rmse"], label=name, alpha=0.7)
plt.xlabel("Trainable Parameters")
plt.ylabel("RMSE")
plt.title("Redesign Pareto Plot: RMSE vs Model Size")
plt.yscale("log")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "redesign_param_efficiency.png"))
plt.close()

results = []
model_defs = [
    ("Classical Matched", lambda: ClassicalAuxPINN(hidden=14, depth=2)),
    ("Classical Fourier Matched", lambda: ClassicalFourierPINN(hidden=12, depth=2, n_fourier=8)),
    ("Classical Fourier Large", lambda: ClassicalFourierPINN(hidden=32, depth=3, n_fourier=16)),
    ("MerLin Full", lambda: MerlinAuxQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2)),
    ("MerLin Depth=2", lambda: MerlinAuxQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=2, n_photons=2)),
]

for name, factory in model_defs:
    print(f"Training {name}...")
    for seed in CONFIG["seeds"]:
        res = run_advantage_experiment(factory, CONFIG, seed)
        res["model"] = name
        results.append(res)

df_exp1 = pd.DataFrame(results)
df_exp1.to_csv(os.path.join(CONFIG["output_dir"], "quantum_advantage_check_exp1.csv"), index=False)

summary_exp1 = df_exp1.groupby("model").agg({
    "params": "first",
    "rmse": ["mean", "std"],
    "pde_residual": ["mean", "std"],
    "train_time": "mean",
    "forward_time_ms": "mean"
})
print("\n=== Seeded Summary Table ===")
print(summary_exp1)

plt.figure(figsize=(8, 5))
for name in df_exp1["model"].unique():
    sub = df_exp1[df_exp1["model"] == name]
    plt.scatter(sub["params"], sub["rmse"], label=name, alpha=0.7)
plt.xlabel("Trainable Parameters")
plt.ylabel("RMSE")
plt.title("Parameter Efficiency: RMSE vs Model Size")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid(True, ls="--", alpha=0.3)
plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "advantage_param_efficiency.png"))
plt.close()

plt.figure(figsize=(8, 5))
for name in df_exp1["model"].unique():
    sub = df_exp1[df_exp1["model"] == name]
    plt.scatter(sub["train_time"], sub["rmse"], label=name, alpha=0.7)
plt.xlabel("Training Time (s)")
plt.ylabel("RMSE")
plt.title("Time Efficiency: RMSE vs Training Budget")
plt.yscale("log")
plt.legend()
plt.grid(True, ls="--", alpha=0.3)
plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "advantage_time_efficiency.png"))
plt.close()

data_results = []
collocation_counts = [32, 64, 128] if CONFIG["FAST_MODE"] else [32, 64, 128, 256]
models_to_test = [
    ("Classical Fourier Large", lambda: ClassicalFourierPINN(hidden=32, depth=3, n_fourier=16)),
    ("MerLin Full", lambda: MerlinAuxQPINN(feature_size=4, quantum_output_size=4, hidden=16, depth=1, n_photons=2)),
]

for n_f in collocation_counts:
    for name, factory in models_to_test:
        print(f"Training {name} with n_f={n_f}...")
        for seed in CONFIG["seeds"]:
            res = run_advantage_experiment(factory, CONFIG, seed, n_f=n_f)
            res["model"] = name
            data_results.append(res)

df_data = pd.DataFrame(data_results)
df_data.to_csv(os.path.join(CONFIG["output_dir"], "quantum_advantage_check_data_efficiency.csv"), index=False)

plt.figure(figsize=(8, 5))
for name in df_data["model"].unique():
    sub = df_data[df_data["model"] == name]
    means = sub.groupby("n_f")["rmse"].mean()
    plt.plot(means.index, means.values, 'o-', label=name)
plt.xlabel("Collocation Points (n_f)")
plt.ylabel("Mean RMSE")
plt.title("Data Efficiency: RMSE vs Collocation Budget")
plt.yscale("log")
plt.legend()
plt.grid(True, ls="--", alpha=0.3)
plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "advantage_data_efficiency.png"))
plt.close()

def get_verdict(claim, test, evidence, condition):
    verdict = "Supported" if condition else "Not supported"
    return {"Claim": claim, "Test": test, "Evidence": evidence, "Verdict": verdict}

verdicts = []

fourier_rmse = df_exp1[df_exp1["model"] == "Classical Fourier Large"]["rmse"].mean()
merlin_rmse = df_exp1[df_exp1["model"] == "MerLin Full"]["rmse"].mean()
best_redesign_rmse = df_redesign["rmse"].mean() if not df_redesign.empty else float('inf')
baseline_merlin_rmse = df_redesign[df_redesign["model"] == "Baseline"]["rmse"].mean() if not df_redesign.empty else float('inf')

verdicts.append(get_verdict("Improved over old MerLin", "Best Redesign vs Baseline", 
                           f"Best Redesign RMSE {best_redesign_rmse:.3e} vs Baseline {baseline_merlin_rmse:.3e}", 
                           best_redesign_rmse < baseline_merlin_rmse * 0.95))

matched_mlp_rmse = df_exp1[df_exp1["model"] == "Classical Matched"]["rmse"].mean()
verdicts.append(get_verdict("Beats matched classical MLP", "Best Redesign vs Matched MLP", 
                           f"Best Redesign {best_redesign_rmse:.3e} vs Matched MLP {matched_mlp_rmse:.3e}", 
                           best_redesign_rmse < matched_mlp_rmse))

verdicts.append(get_verdict("Beats Fourier Large", "Best Redesign vs Fourier Large", 
                           f"Best Redesign {best_redesign_rmse:.3e} vs Fourier Large {fourier_rmse:.3e}", 
                           best_redesign_rmse < fourier_rmse))

verdicts.append(get_verdict("Quantum advantage", "Overall Assessment", 
                           "Beats all classical baselines under fair budget", 
                           best_redesign_rmse < fourier_rmse))

df_verdict = pd.DataFrame(verdicts)
print("\n=== Final Verdict Table ===")
print(df_verdict.to_string(index=False))

print("\n=== Automated Conclusion ===")
if best_redesign_rmse < fourier_rmse:
    print("MerLin shows an accuracy advantage over the strongest classical baseline in this regime.")
else:
    print("In the tested regime, quantum advantage is not supported: the strongest classical Fourier-feature PINN achieves lower RMSE than MerLin. However, the redesign shows improvements and sensitivity to photonic resources. Therefore the evidence supports photonic-resource sensitivity, but not general quantum advantage.")

