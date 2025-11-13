"""
Figure 1 — Load-Following Introduction (Publication Grade)

This script builds a deterministic fleet of agents, computes the aggregated
system baseline load, and defines a new target load shape that is close to the
baseline but slightly modified. It then renders a two-row publication-grade
figure:

- Top: System Baseline (dashed) vs Target Shape (solid red)
- Bottom: Absolute error |Baseline − Target| as a filled red area

The random seed is fixed for full reproducibility.
"""

from __future__ import annotations

import os
import sys
import random
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Toggle heuristic vs exact agent implementations
USE_HEURISTIC = False

if USE_HEURISTIC:
    from faa_agents_heuristic import (
        BuildingHVACAgentHeuristic as BuildingHVACAgent,
        EnergyStorageAgentHeuristicFast as EnergyStorageAgent,
        CILoadShiftAgentHeuristicFast as CILoadShiftAgent,
    )
else:
    from faa_agents import BuildingHVACAgent, EnergyStorageAgent, CILoadShiftAgent
from market_operator import MarketOperator


# -----------------------------
# Reproducibility and parameters
# -----------------------------
SEED = 20251102
N_TIMESTEPS = 24
N_AGENTS_PER_TYPE = 50  # publication default


def _rng_reset(seed: int = SEED) -> Tuple[random.Random, np.random.Generator]:
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed), np.random.default_rng(seed)


def build_environment(T: int) -> dict:
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    # Add a small evening spike to make the shape interesting
    prices[int(0.7 * T) : min(T, int(0.7 * T) + 3)] *= 1.4
    return {
        "outdoor_temp_c": 10 + 10 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 1.5),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }


def build_fleet(T: int, n_per_type: int, rng_py: random.Random):
    agents = []
    # HVAC
    for i in range(n_per_type):
        agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=float(rng_py.uniform(5, 15)),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=float(rng_py.uniform(0.4, 1.0)),
            )
        )
    # Storage
    for i in range(n_per_type):
        kwargs = dict(
            capacity_kwh=float(rng_py.uniform(10, 50)),
            max_power_kw=float(rng_py.uniform(2, 10)),
            cycle_cost_dol_per_kwh=float(rng_py.uniform(0.02, 0.08)),
        )
        if USE_HEURISTIC:
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs, lookups=None))
        else:
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs))
    # C&I with randomized window to avoid clustering
    for i in range(n_per_type):
        duration = int(rng_py.randint(2, 6))
        lo = int(rng_py.randint(0, max(0, T - duration)))
        hi_min = lo + duration - 1
        hi_max = T - 1
        hi = int(rng_py.randint(hi_min, min(hi_min + 8, hi_max)))
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng_py.uniform(50, 150)),
                duration_hours=duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=float(rng_py.uniform(75, 200)),
            )
        )
    return agents


def collect_system_baseline_and_bids(agents, sim_state):
    T = len(sim_state["price_dol_per_kwh"])
    baseline = np.zeros(T, dtype=float)
    all_bids = []
    for ag in agents:
        baseline += ag.get_baseline_operation(sim_state)
        # Use zero premium (no markup) for publication reproducibility
        all_bids.extend(ag.get_flexibility_bids(sim_state))
    return baseline, all_bids


def make_target_from_baseline(baseline: np.ndarray) -> np.ndarray:
    """
    Construct a target that follows the baseline everywhere except:
    - Hours 4, 5, 6 set to 90% of baseline at hour 4
    - Hours 19, 20, 21 set to 90% of baseline at hour 19
    - Hours 11, 12, 13 set to 110% of baseline at hour 11
    """
    T = len(baseline)
    target = baseline.copy()

    def set_from_ref(start: int, end_incl: int, ref: int, scale: float) -> None:
        if T == 0:
            return
        s = max(0, start)
        e = min(T - 1, end_incl)
        if e < s:
            s, e = e, s
        ref_idx = int(np.clip(ref, 0, T - 1))
        val = float(baseline[ref_idx]) * scale
        for i in range(s, e + 1):
            target[i] = max(0.0, val)

    set_from_ref(4, 7, 4, 0.90)
    set_from_ref(19, 22, 19, 0.90)
    set_from_ref(11, 13, 11, 1.10)
    return target


def render_figure(
    baseline: np.ndarray, target: np.ndarray, achieved: np.ndarray, out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    x = list(range(len(baseline)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.80, 0.20],
    )

    # Row 1 — Baseline (dashed), Target (solid red), Achieved (solid black)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=baseline,
            name="System Baseline",
            line=dict(color="#7f7f7f", dash="dash"),
        ),
        row=1,
        col=1,
    )
    # Draw achieved first, then target last so gold sits on top
    fig.add_trace(
        go.Scatter(
            x=x, y=achieved, name="Achieved Load", line=dict(color="black", width=3)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=target, name="Load Target", line=dict(color="#DAA520", width=3)
        ),
        row=1,
        col=1,
    )

    # Row 2 — Absolute error area (red fill) between Achieved and Target
    err = np.abs(achieved - target)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=err,
            name="|Achieved − Target|",
            line=dict(color="#d62728"),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.35)",
        ),
        row=2,
        col=1,
    )

    # Styling
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Error (kW)", row=2, col=1, nticks=3)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    # Save publication-grade outputs
    pdf_path = os.path.join(out_dir, "figure_01_load_following_intro.pdf")
    png_path = os.path.join(out_dir, "figure_01_load_following_intro.png")
    try:
        fig.write_image(pdf_path, format="pdf")
        fig.write_image(png_path, format="png", scale=2)
        print(f"Saved figure to: {pdf_path} and {png_path}")
    except Exception as exc:
        print(
            f"Warning: could not save static images ({exc}). Showing interactive figure instead."
        )
        fig.show()


def main():
    rng_py, _ = _rng_reset(SEED)
    sim = build_environment(N_TIMESTEPS)
    fleet = build_fleet(N_TIMESTEPS, N_AGENTS_PER_TYPE, rng_py)
    # If heuristic storage is used, populate lookups once using prices
    if USE_HEURISTIC:
        try:
            from faa_agents_heuristic import EnergyStorageAgentHeuristicFast

            lookups = EnergyStorageAgentHeuristicFast.precompute_lookups(
                sim["price_dol_per_kwh"]
            )
            for ag in fleet:
                if hasattr(ag, "agent_type") and getattr(ag, "agent_type") == "Storage":
                    if hasattr(ag, "_lookups"):
                        ag._lookups = lookups
        except Exception:
            pass
    baseline, all_bids = collect_system_baseline_and_bids(fleet, sim)
    target = make_target_from_baseline(baseline)
    # Run market clearing to follow the target shape across the full horizon
    mo = MarketOperator(peak_guard_factor=1.05)
    achieved, dispatched, total_cost = mo.clear_market(
        all_bids=all_bids,
        system_baseline_load=baseline,
        target_load_shape=target,
    )
    print(
        f"Clearing summary: dispatched {len(dispatched)} bids | total cost ${total_cost:.2f}"
    )
    # Save under project-level results/ directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_root, "results")
    render_figure(baseline, target, achieved, out_dir=out_dir)


if __name__ == "__main__":
    main()
