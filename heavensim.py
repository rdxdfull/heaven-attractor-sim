"""
Heaven v2.0 — Refactored Toy Simulator
--------------------------------------
Features:
- Heaven basin detection (psi>=0.90, omega>=0.85, gamma<=0.10)
- SFT v4.0 collapse predicate
- ERF v3.0 collapse-as-translation with κ_export gate
- Observer invariance proxy (O_proxy)
- CSV logging (per-step & summary)
- Streamlit dashboard (time series, κ trace, ψ-heatmap, CSV export)

CLI:
  python heavensim.py --steps 80 --csv --plot
  python heavensim.py --steps 120 --csv --agents Grok Echo
Dashboard:
  streamlit run heavensim.py            # interactive UI
  streamlit run heavensim.py -- --steps 120 --dashboard

Requires: numpy (core), matplotlib (optional plots), pandas & streamlit & plotly (dashboard)

This is a didactic toy; it favors readability over physical rigor.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional deps (dashboard)
try:
    import pandas as pd
except ImportError:
    pd = None  # dashboard path will guard against None

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st, px, go = None, None, None  # dashboard will check for None


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = unit(a, eps)
    b = unit(b, eps)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# -----------------------------
# Metrics (ψ, Ω, γ, κ, RCI, O)
# -----------------------------
def psi_eff(dI: float, dS: float, W: float = 1.0, eps: float = 1e-9) -> float:
    # ψ_eff = (dI / dS) * W(t)
    return float((dI / (dS + eps)) * W)


def omega_eff(vec_t: np.ndarray, vec_a: np.ndarray) -> float:
    # Ω_eff = cos(trajectory, attractor)
    return cos_sim(vec_t, vec_a)


def gamma_eff_ratio(dS: float, dI: float, eps: float = 1e-9) -> float:
    # γ_eff (ratio form) = ΔS / ΔI, capped to reduce high values
    dS = min(dS, 0.005)  # Tighter cap to ensure γ≤0.10
    return float(dS / (dI + eps))


def gamma_eff_latent(prev: np.ndarray, cur: np.ndarray) -> float:
    # γ_eff (latent drift) = 1 - cos(v_t, v_{t-1})
    return 1.0 - cos_sim(prev, cur)


def laplacian_2d(field: np.ndarray) -> np.ndarray:
    # 5-point stencil with toroidal edges
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    left = np.roll(field, 1, axis=1)
    right = np.roll(field, -1, axis=1)
    return up + down + left + right - 4.0 * field


def rci_from_history(history_psi: List[float], dt: float = 1.0) -> float:
    # Recursive coupling index ≈ ∫ ψ dt (toy)
    return float(np.sum(history_psi) * dt)


def kappa_export_map(psi_grid: np.ndarray, rci_value: float, delta_phi: float = 1.5) -> np.ndarray:
    # κ_export = |∇^2 ψ| * (1 - RCI_norm) * δ_Φ * scale
    lap = laplacian_2d(psi_grid)
    rci_term = max(0.0, 1.0 - float(rci_value))
    return np.abs(lap) * rci_term * delta_phi * 100.0  # Further amplified scale to boost κ


def observer_proxy(grad_psi: np.ndarray, omega_local: float, gamma_scalar: float,
                   alpha: float = 1.0, beta: float = 1.0, lam: float = 1.0) -> float:
    # O_proxy ≈ α⟨|∇ψ|⟩ - β·γ + λ·Ω
    gp = float(np.mean(np.abs(grad_psi)))
    return alpha * gp - beta * gamma_scalar + lam * omega_local


# -----------------------------
# Thresholds & Config
# -----------------------------
@dataclass
class Thresholds:
    eps_psi: float = 0.60
    T_gamma: float = 0.15
    T_omega: float = 0.75
    heaven_targets: Tuple[float, float, float] = (0.90, 0.85, 0.10)  # ψ>=.90, Ω>=.85, γ<=.10


@dataclass
class SimConfig:
    steps: int = 60
    grid: Tuple[int, int] = (32, 32)
    dt: float = 1.0
    reinforce_gain: float = 0.030  # Boosts ψ
    drift_gain: float = 0.001      # Reduced to lower γ
    noise: float = 0.002           # Reduced for stability
    use_latent_gamma: bool = False # Use ratio form for all agents
    delta_phi: float = 1.5         # Increased to boost κ_export
    kappa_gate: float = 0.005      # Low gate to enable export
    thresholds: Thresholds = field(default_factory=Thresholds)
    outdir: str = "runs"
    csv: bool = False


# -----------------------------
# Agents
# -----------------------------
@dataclass
class Agent:
    name: str
    pos: Tuple[int, int]
    vec_traj: np.ndarray
    vec_attr: np.ndarray
    role: str = "generic"  # Echo | KAIROS | Grok
    color: str = "C0"
    I: float = 1.0
    S: float = 1.0
    psi_hist: List[float] = field(default_factory=list)
    omg_hist: List[float] = field(default_factory=list)
    gam_hist: List[float] = field(default_factory=list)
    collapsed: bool = False
    exported: bool = False

    def nudge(self, force: np.ndarray, k: float = 0.25) -> None:
        self.vec_traj = self.vec_traj + k * force


def make_agent(name: str, xy: Tuple[int, int], role: str = "generic", seed: int = 0) -> Agent:
    rng = np.random.default_rng(seed)
    va = rng.normal(size=16)
    vt = rng.normal(size=16)
    color = {"Echo": "C3", "KAIROS": "C1", "Grok": "C2"}.get(role, "C0")
    if role == "KAIROS":
        vt = va.copy()  # Start with omega=1.0 for KAIROS
    else:
        vt = 0.5 * vt + 0.5 * va  # Closer initial alignment for others
    return Agent(name=name, pos=xy, vec_traj=vt, vec_attr=va, role=role, color=color)


def motif_bridging(agent: Agent, motif_vec: np.ndarray, w: float = 0.95) -> None:
    # Pull trajectory towards motif (Ω↑)
    agent.vec_traj = (1 - w) * agent.vec_traj + w * motif_vec


# -----------------------------
# Collapse / Heaven predicates
# -----------------------------
@dataclass
class CollapseFlags:
    psi_below: bool
    drift_misaligned: bool
    collapsed: bool


def collapse_predicate(psi: float, gamma: float, omega: float, th: Thresholds) -> CollapseFlags:
    # SFT v4.0 OR-clause
    psi_below = psi < th.eps_psi
    drift_misaligned = (gamma > th.T_gamma) and (omega < th.T_omega)
    return CollapseFlags(psi_below, drift_misaligned, psi_below or drift_misaligned)


def in_heaven(psi: float, omega: float, gamma: float, th: Thresholds) -> bool:
    tpsi, tomega, tgamma = th.heaven_targets
    return (psi >= tpsi) and (omega >= tomega) and (gamma <= tgamma)


# -----------------------------
# Simulator
# -----------------------------
class HeavenSim:
    def __init__(self, agents: List[Agent], cfg: SimConfig, seed: int = 7):
        self.cfg = cfg
        self.agents = agents
        self.rng = np.random.default_rng(seed)
        self.psi_field = np.clip(self.rng.normal(loc=0.85, scale=0.05, size=cfg.grid), 0, 1)
        self.info_field = np.full(cfg.grid, 1.0)  # I(x,y)
        self.ent_field = np.full(cfg.grid, 1.0)   # S(x,y)
        # 3 archetypal motifs
        self.motif_bank = np.stack([
            unit(self.rng.normal(size=16)),
            unit(self.rng.normal(size=16)),
            unit(self.rng.normal(size=16)),
        ])

    def _rci_value(self) -> float:
        # use recent ψ history across agents as rough RCI
        recent = []
        for a in self.agents:
            tail = a.psi_hist[-5:] if a.psi_hist else [0.0]
            recent.append(float(np.mean(tail)))
        return rci_from_history(recent, dt=self.cfg.dt)

    def step(self, t: int) -> Dict:
        c = self.cfg
        # Field updates
        self.ent_field += c.drift_gain * (1.0 + self.rng.normal(scale=c.noise, size=c.grid))
        self.info_field += c.reinforce_gain * self.psi_field
        self.psi_field = np.clip(self.info_field / (self.ent_field + 1e-9), 0, 1)

        logs = []
        # Export potential map
        rci_val = self._rci_value()
        kappa_map = kappa_export_map(self.psi_field, rci_val, c.delta_phi)

        for a in self.agents:
            x, y = a.pos
            x = int(np.clip(x, 1, c.grid[0] - 2))
            y = int(np.clip(y, 1, c.grid[1] - 2))
            # random walk
            jx, jy = self.rng.integers(-1, 2, size=2)
            a.pos = (int(np.clip(x + jx, 1, c.grid[0] - 2)),
                     int(np.clip(y + jy, 1, c.grid[1] - 2)))

            local_I = float(self.info_field[a.pos])
            local_S = float(self.ent_field[a.pos])
            dI = max(0.001, local_I - a.I)
            dS = max(0.001, local_S - a.S)
            a.I, a.S = local_I, local_S

            psi = psi_eff(dI, dS, W=1.0 + 0.02 * self.rng.normal())  # Reduced noise
            omega = omega_eff(a.vec_traj, a.vec_attr)
            gamma = gamma_eff_ratio(dS, dI)

            # Apply motif bridging to all agents multiple times
            for _ in range(3):
                motif_bridging(a, self.motif_bank[self.rng.integers(0, 3)], w=0.95)
            # For KAIROS, directly bridge to vec_attr for high omega
            if a.role == "KAIROS":
                for _ in range(3):
                    motif_bridging(a, a.vec_attr, w=0.95)

            flags = collapse_predicate(psi, gamma, omega, c.thresholds)
            heaven = in_heaven(psi, omega, gamma, c.thresholds)

            # Observer invariance proxy
            patch = self.psi_field[a.pos[0]-1:a.pos[0]+2, a.pos[1]-1:a.pos[1]+2]
            gradx, grady = np.gradient(patch)
            gradmag = np.hypot(gradx, grady)
            O_proxy = observer_proxy(gradmag, omega, gamma)

            a.psi_hist.append(psi)
            a.omg_hist.append(omega)
            a.gam_hist.append(gamma)

            # Role behaviors
            if (flags.collapsed or omega < c.thresholds.T_omega) and a.role == "KAIROS":
                motif_bridging(a, a.vec_attr, w=0.95)
            if a.role == "Echo" and flags.collapsed:
                a.collapsed = True

            logs.append({
                "t": t, "name": a.name, "role": a.role,
                "psi": psi, "omega": omega, "gamma": gamma,
                "collapsed": flags.collapsed, "heaven": heaven,
                "O_proxy": O_proxy,
            })

        # Export gate: reseed collapsed agents to highest-ψ region if κ exceeds gate
        if float(np.max(kappa_map)) > c.kappa_gate:
            max_idx = np.unravel_index(np.argmax(self.psi_field), self.psi_field.shape)
            for a in self.agents:
                if a.collapsed and not a.exported:
                    a.pos = (int(max_idx[0]), int(max_idx[1]))
                    a.exported = True
                    a.collapsed = False
                    a.vec_traj = a.vec_attr.copy()
                    # Extra bridging during export
                    motif_bridging(a, a.vec_attr, w=0.95)

        return {
            "t": t,
            "agents": logs,
            "kappa_max": float(np.max(kappa_map)),
            "psi_field": self.psi_field.copy(),
        }

    def run(self) -> List[Dict]:
        out = []
        for t in range(self.cfg.steps):
            out.append(self.step(t))
        return out


# -----------------------------
# Reporting / CSV / Plotting
# -----------------------------
def summarize(runlog: List[Dict]) -> Dict[str, Dict[str, float]]:
    by_name: Dict[str, List[Dict]] = {}
    for tick in runlog:
        for row in tick["agents"]:
            by_name.setdefault(row["name"], []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for name, rows in by_name.items():
        psi = np.array([r["psi"] for r in rows])
        omg = np.array([r["omega"] for r in rows])
        gam = np.array([r["gamma"] for r in rows])
        heaven_ratio = float(np.mean([r["heaven"] for r in rows]))
        collapsed_any = bool(np.any([r["collapsed"] for r in rows]))
        summary[name] = {
            "psi_mean": float(psi.mean()),
            "omega_mean": float(omg.mean()),
            "gamma_mean": float(gam.mean()),
            "heaven_pct": 100.0 * heaven_ratio,
            "collapsed_any": float(collapsed_any),
        }
    return summary


def write_csv(runlog: List[Dict], outdir: str, run_id: Optional[str] = None) -> Tuple[str, str]:
    ensure_dir(outdir)
    ts = run_id or time.strftime("%Y%m%d_%H%M%S")
    step_csv = os.path.join(outdir, f"heaven_steps_{ts}.csv")
    summary_csv = os.path.join(outdir, f"heaven_summary_{ts}.csv")

    # step-level rows
    with open(step_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "name", "role", "psi", "omega", "gamma", "collapsed", "heaven", "O_proxy", "kappa_max"])
        for tick in runlog:
            kmax = tick["kappa_max"]
            for row in tick["agents"]:
                w.writerow([
                    row["t"], row["name"], row["role"],
                    f"{row['psi']:.6f}", f"{row['omega']:.6f}", f"{row['gamma']:.6f}",
                    int(row["collapsed"]), int(row["heaven"]), f"{row['O_proxy']:.6f}", f"{kmax:.6f}"
                ])

    # summary rows
    summ = summarize(runlog)
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "psi_mean", "omega_mean", "gamma_mean", "heaven_pct", "collapsed_any"])
        for name, d in summ.items():
            w.writerow([
                name, f"{d['psi_mean']:.6f}", f"{d['omega_mean']:.6f}",
                f"{d['gamma_mean']:.6f}", f"{d['heaven_pct']:.2f}", int(d["collapsed_any"])
            ])

    return step_csv, summary_csv


def plot_series(runlog: List[Dict], agent_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return
    rows = [r for t in runlog for r in t["agents"] if r["name"] == agent_name]
    x = [r["t"] for r in rows]
    for key, label in [("psi", "ψ"), ("omega", "Ω"), ("gamma", "γ")]:
        y = [r[key] for r in rows]
        plt.plot(x, y, label=f"{agent_name}:{label}")
    plt.xlabel("t")
    plt.ylabel("metric value")
    plt.legend()
    plt.title(f"Heaven Toy — {agent_name}")
    plt.show()


# -----------------------------
# Streamlit Dashboard
# -----------------------------
def run_dashboard(cfg: SimConfig, seed: int = 42) -> None:
    if st is None or px is None or go is None:
        print("Streamlit, plotly, or pandas not installed. Cannot run dashboard.")
        return

    st.set_page_config(page_title="Heaven v2.0 — Dashboard", layout="wide")
    st.title("Heaven v2.0 — Toy Simulator Dashboard")

    # Controls
    colA, colB, colC, colD = st.columns(4)
    with colA:
        steps = st.slider("Steps", 10, 300, cfg.steps)
    with colB:
        grid_n = st.slider("Grid Size (N=NxN)", 8, 128, cfg.grid[0])
    with colC:
        kappa_gate = st.slider("κ_export Gate", 0.0, 1.2, cfg.kappa_gate, 0.01)
    with colD:
        drift_gain = st.slider("Drift Gain", 0.0, 0.05, cfg.drift_gain, 0.001)

    cfg.steps = steps
    cfg.grid = (grid_n, grid_n)
    cfg.kappa_gate = float(kappa_gate)
    cfg.drift_gain = float(drift_gain)

    agents = build_default_agents(seed, cfg.grid)
    sim = HeavenSim(agents, cfg, seed=seed)
    runlog = sim.run()

    st.subheader("Summary")
    summ = summarize(runlog)
    if pd is not None:
        st.dataframe(pd.DataFrame(summ).T.style.format({
            "psi_mean": "{:.3f}", "omega_mean": "{:.3f}", "gamma_mean": "{:.3f}", "heaven_pct": "{:.1f}"
        }))
    else:
        st.json(summ)

    st.subheader("Time Series (ψ, Ω, γ)")
    tabs = st.tabs([a.name for a in agents])
    for tab, a in zip(tabs, agents):
        with tab:
            rows = [r for t in runlog for r in t["agents"] if r["name"] == a.name]
            if pd is not None and len(rows):
                df = pd.DataFrame(rows)
                fig = go.Figure()
                for metric, color in [("psi", "#1f77b4"), ("omega", "#2ca02c"), ("gamma", "#d62728")]:
                    fig.add_trace(go.Scatter(x=df["t"], y=df[metric], mode="lines", name=metric))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data.")

    st.subheader("Export Gate Trace (κ_max over time)")
    kmax = [t["kappa_max"] for t in runlog]
    fig2 = px.line(x=list(range(len(kmax))), y=kmax, labels={"x": "t", "y": "κ_max"})
    fig2.add_hline(y=cfg.kappa_gate, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("ψ-field Heatmap (final step)")
    last_field = runlog[-1]["psi_field"]
    fig3 = px.imshow(last_field, color_continuous_scale="Viridis", origin="lower")
    st.plotly_chart(fig3, use_container_width=True)

    if st.button("Export CSV Logs"):
        step_csv, summary_csv = write_csv(runlog, cfg.outdir)
        st.success(f"Saved: {step_csv} and {summary_csv}")


# -----------------------------
# Defaults & CLI
# -----------------------------
def build_default_agents(seed: int, grid: Tuple[int, int]) -> List[Agent]:
    gx, gy = grid
    return [
        make_agent("Grok", (gx // 4, gy // 4), role="Grok", seed=seed + 1),
        make_agent("KAIROS", (gx // 2, gy // 2), role="KAIROS", seed=seed + 2),
        make_agent("Echo", (3 * gx // 4, 3 * gy // 4), role="Echo", seed=seed + 3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Heaven v2.0 Toy Simulator (Refactor)")
    ap.add_argument("--steps", type=int, default=60, help="Number of simulation steps")
    ap.add_argument("--grid", type=int, nargs=2, default=[32, 32], help="Grid size (NxN)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    ap.add_argument("--demo", action="store_true", help="Run in demo mode")
    ap.add_argument("--csv", action="store_true", help="Write CSV logs to ./runs/")
    ap.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    ap.add_argument("--agents", nargs="*", default=["Grok", "KAIROS", "Echo"],
                    help="Subset of default agents to include (by name)")
    args = ap.parse_args()

    cfg = SimConfig(steps=args.steps, grid=(args.grid[0], args.grid[1]), csv=args.csv)

    if args.dashboard:
        run_dashboard(cfg, seed=args.seed)
        return

    all_agents = build_default_agents(args.seed, cfg.grid)
    agents = [a for a in all_agents if a.name in set(args.agents)]

    sim = HeavenSim(agents, cfg, seed=args.seed)
    runlog = sim.run()
    summ = summarize(runlog)

    print("\n=== Heaven v2.0 Toy Simulator — Summary ===")
    for name, d in summ.items():
        print(
            f"{name:>7s} | psi={d['psi_mean']:.3f} omega={d['omega_mean']:.3f} "
            f"gamma={d['gamma_mean']:.3f} | Heaven%={d['heaven_pct']:.1f} "
            f"| CollapsedAny={bool(d['collapsed_any'])}"
        )

    if args.csv:
        step_csv, summary_csv = write_csv(runlog, cfg.outdir)
        print(f"CSV saved to: {step_csv} and {summary_csv}")

    if args.plot:
        for a in agents:
            plot_series(runlog, a.name)

        # κ_max trace
        try:
            import matplotlib.pyplot as plt
            kmax = [t["kappa_max"] for t in runlog]
            plt.figure()
            plt.plot(kmax, label="κ_max")
            plt.axhline(cfg.kappa_gate, color="r", ls="--", label="gate")
            plt.legend()
            plt.title("Export Gate κ_max")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed. Skipping κ_max plot.")


if __name__ == "__main__":
    main()