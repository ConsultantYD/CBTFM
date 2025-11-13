"""
Figure 4 — Grid Search: Event Load Following (Publication Grade)

Build a single environment and a maximum fleet pool, then sweep a grid of
absolute resource counts (0..100 by 10) for Storage and C&I while keeping HVAC
fixed. For each grid point, clear the market for a 3-hour 20% shed (17–19)
and record:

- Left heatmap: Mean Absolute Error (MAE, MW) between Achieved and Target in
  the event window (darker reds = higher error).
- Right heatmap: Normalized total cost across the grid (min–max scaled 0..1).

For fast iteration, HVAC is set to 10 by default. For publication, increase to
100. Storage and C&I counts vary directly as absolute numbers.
"""

from __future__ import annotations

import os
import sys
import random
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

USE_HEURISTIC = True

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
# Parameters (tweak for publication)
# -----------------------------
SEED = 20251102
N_TIMESTEPS = 24

# Fixed HVAC count (set to 100 for publication)
N_HVAC_FIXED = 100  # set to 100 for final figure

# Grid counts (0..100 in steps of 10)
COUNT_GRID = list(range(0, 101, 5))

# Event window: 3h 20% shed (17–19)
EVENT_START = 17
EVENT_END = 19

# Outdoor temperature shift to make the event more demanding (colder -> more HVAC)
OUTDOOR_TEMP_SHIFT_C = -3.0


def _rng_reset(seed: int = SEED) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def build_env(T: int) -> dict:
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    prices[int(0.7 * T) : min(T, int(0.7 * T) + 3)] *= 1.5
    outdoor = 10 + 10 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 1.6)
    outdoor = outdoor + float(OUTDOOR_TEMP_SHIFT_C)
    return {
        "outdoor_temp_c": outdoor,
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }


def target_from_window_cap(
    baseline: np.ndarray, s: int, e: int, cap_kw: float
) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = np.minimum(baseline[window], cap_kw)
    return tgt


def target_from_window_fraction(
    baseline: np.ndarray, s: int, e: int, shed_fraction: float
) -> np.ndarray:
    """Reduce baseline by a fixed fraction within [s, e] inclusive.

    shed_fraction is in [0,1]. For example, 0.5 means target = 50% of baseline
    in the window.
    """
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = baseline[window] * (1.0 - float(shed_fraction))
    return tgt


def build_fleet_pool(T: int, rng: random.Random, sim: dict) -> List:
    """Construct a maximum pool of agents we can subset for each grid point.

    - HVAC: fixed N_HVAC_FIXED
    - Storage/C&I: up to 200 absolute assets
    """
    agents: List = []

    # HVAC pool (fixed)
    for i in range(N_HVAC_FIXED):
        agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=float(rng.uniform(5, 15)),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=float(rng.uniform(0.4, 1.0)),
            )
        )

    # Storage pool up to 200 absolute assets
    max_storage = int(max(COUNT_GRID))
    for i in range(max_storage):
        agents.append(
            EnergyStorageAgent(
                f"bess_{i}",
                capacity_kwh=float(rng.uniform(10, 50)),
                max_power_kw=float(rng.uniform(2, 10)),
                cycle_cost_dol_per_kwh=float(rng.uniform(0.02, 0.08)),
                lookups=None if USE_HEURISTIC else None,
            )
        )

    # C&I pool up to 200 absolute assets
    max_ci = int(max(COUNT_GRID))
    for i in range(max_ci):
        duration = int(rng.randint(2, 6))
        lo = int(rng.randint(0, max(0, T - duration)))
        hi_min = lo + duration - 1
        hi_max = T - 1
        hi = int(rng.randint(hi_min, min(hi_min + 8, hi_max)))
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng.uniform(50, 150)),
                duration_hours=duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=float(rng.uniform(75, 200)),
            )
        )

    # If heuristic storage is used, share price lookups
    if USE_HEURISTIC:
        try:
            from faa_agents_heuristic import EnergyStorageAgentHeuristicFast as _ESS

            lookups = _ESS.precompute_lookups(sim["price_dol_per_kwh"])
            for ag in agents:
                if getattr(ag, "agent_type", "") == "Storage" and hasattr(
                    ag, "_lookups"
                ):
                    ag._lookups = lookups
        except Exception:
            pass

    return agents


def precompute_agent_baselines_and_bids(
    agents: List, sim: dict
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict]]]:
    baselines: Dict[str, np.ndarray] = {}
    bids: Dict[str, List[Dict]] = {}
    for ag in agents:
        aid = getattr(ag, "agent_id")
        baselines[aid] = np.asarray(ag.get_baseline_operation(sim), dtype=float)
        bids[aid] = ag.get_flexibility_bids(sim)
    return baselines, bids


def select_agents(agents: List, n_storage: int, n_ci: int) -> List:
    sel: List = []
    # Include all HVAC (first N_HVAC_FIXED)
    hvac = [ag for ag in agents if getattr(ag, "agent_type", "") == "HVAC"][
        :N_HVAC_FIXED
    ]
    sel.extend(hvac)
    # Select storage subset in order
    storage = [ag for ag in agents if getattr(ag, "agent_type", "") == "Storage"][
        : max(0, int(n_storage))
    ]
    sel.extend(storage)
    # Select C&I subset in order
    ci = [ag for ag in agents if getattr(ag, "agent_type", "") == "C&I"][
        : max(0, int(n_ci))
    ]
    sel.extend(ci)
    return sel


def assemble_system_baseline_and_bids(
    selected: List,
    baselines_by_id: Dict[str, np.ndarray],
    bids_by_id: Dict[str, List[Dict]],
) -> Tuple[np.ndarray, List[Dict]]:
    # Sum baselines
    T = int(len(next(iter(baselines_by_id.values())))) if baselines_by_id else 0
    system_baseline = np.zeros(T, dtype=float)
    for ag in selected:
        aid = getattr(ag, "agent_id")
        system_baseline += np.asarray(baselines_by_id[aid], dtype=float)
    # Collect bids
    all_bids: List[Dict] = []
    for ag in selected:
        aid = getattr(ag, "agent_id")
        all_bids.extend(bids_by_id[aid])
    return system_baseline, all_bids


def render_heatmaps(
    count_grid: List[int],
    z_mae_mw: np.ndarray,
    z_cost_norm: np.ndarray,
    mask_black: np.ndarray,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    x = count_grid
    y = count_grid

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("MAE in Event Window (MW)", "Normalized Cost (0–1)"),
    )

    # Left: MAE heatmap
    fig.add_trace(
        go.Heatmap(
            z=z_mae_mw,
            x=x,
            y=y,
            colorscale="Reds",
            reversescale=False,
            colorbar=dict(title="MW", x=0.46, len=0.8),
            zsmooth=False,
        ),
        row=1,
        col=1,
    )

    # Right: normalized cost heatmap (green colormap)
    fig.add_trace(
        go.Heatmap(
            z=z_cost_norm,
            x=x,
            y=y,
            colorscale="Greens",
            reversescale=False,
            colorbar=dict(title="Norm. Cost", x=1.02, len=0.8),
            zsmooth=False,
        ),
        row=1,
        col=2,
    )

    # Overlay black mask where MAPE > threshold
    z_mask = np.where(mask_black, 1.0, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=z_mask,
            x=x,
            y=y,
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,1)"]],
            showscale=False,
            zmin=0,
            zmax=1,
        ),
        row=1,
        col=2,
    )

    # Axis labeling
    fig.update_xaxes(title_text="Storage (count)", row=1, col=1)
    fig.update_xaxes(title_text="Storage (count)", row=1, col=2)
    fig.update_yaxes(title_text="C&I (count)", row=1, col=1)

    # Guide ticks at 0, 20, 40, 60, 80, 100
    tickvals = [0, 20, 40, 60, 80, 100]
    for c in (1, 2):
        fig.update_xaxes(tickmode="array", tickvals=tickvals, row=1, col=c)
    fig.update_yaxes(tickmode="array", tickvals=tickvals, row=1, col=1)

    # Make each subplot square (x=y) and adjust colorbars to avoid overlap
    for c in (1, 2):
        fig.update_xaxes(range=[0, 100], row=1, col=c)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    # Square aspect for both panels
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    # Make each subplot square (x=y) and adjust colorbars to avoid overlap
    for c in (1, 2):
        fig.update_xaxes(range=[0, 200], row=1, col=c)
    fig.update_yaxes(range=[0, 200], row=1, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    # No gridlines on axes; cleaner visual
    for c in (1, 2):
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, row=1, col=c)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, row=1, col=1)

    # No main title; rely on subplot titles
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=8, r=8, t=20, b=8),
        autosize=True,
    )

    # Save outputs
    pdf_path = os.path.abspath(
        os.path.join(out_dir, "figure_04_grid_search_event_following.pdf")
    )
    png_path = os.path.abspath(
        os.path.join(out_dir, "figure_04_grid_search_event_following.png")
    )
    # Show in browser and also save static images
    fig.show(config={"responsive": True})
    try:
        fig.write_image(pdf_path, format="pdf")
        fig.write_image(png_path, format="png", scale=2)
        print(f"Saved figure to: {pdf_path} and {png_path}")
    except Exception as exc:
        print(f"Warning: could not save static images ({exc}).")


def main():
    rng = _rng_reset(SEED)
    sim = build_env(N_TIMESTEPS)

    # Build max pool and precompute per-agent baselines and bids
    pool = build_fleet_pool(N_TIMESTEPS, rng, sim)
    baselines_by_id, bids_by_id = precompute_agent_baselines_and_bids(pool, sim)

    # Constant system baseline: sum of ALL pool baselines (max scenario).
    # As we vary counts, non-selected agents are treated as inflexible load
    # (baseline only, no bids), so total baseline stays constant across grid.
    system_baseline_const = np.zeros(N_TIMESTEPS, dtype=float)
    for aid, bl in baselines_by_id.items():
        system_baseline_const += np.asarray(bl, dtype=float)

    # Fixed target from the constant baseline — 20% shed in the event window
    target_const = target_from_window_fraction(
        system_baseline_const, EVENT_START, EVENT_END, 0.2
    )

    # Metric arrays (rows: C&I counts, cols: Storage counts)
    n = len(COUNT_GRID)
    mae_mw = np.zeros((n, n), dtype=float)
    cost = np.zeros((n, n), dtype=float)
    mape_pct = np.zeros((n, n), dtype=float)

    mo = MarketOperator(peak_guard_factor=1.05)

    # For each grid point, assemble subset bids only; baseline remains constant
    for iy, ci_cnt in enumerate(COUNT_GRID):
        n_ci = int(ci_cnt)
        for ix, st_cnt in enumerate(COUNT_GRID):
            n_st = int(st_cnt)

            selected = select_agents(pool, n_storage=n_st, n_ci=n_ci)

            # Build bids for selected agents only (others remain inflexible)
            all_bids: List[Dict] = []
            for ag in selected:
                aid = getattr(ag, "agent_id")
                all_bids.extend(bids_by_id.get(aid, []))

            achieved, dispatched, total_cost = mo.clear_market(
                all_bids=all_bids,
                system_baseline_load=system_baseline_const,
                target_load_shape=target_const,
                focus_windows=[(EVENT_START, EVENT_END)],
                max_bids_per_agent=1,
            )

            # Metrics within the event window
            w = slice(EVENT_START, EVENT_END + 1)
            err_kw = np.abs(achieved[w] - target_const[w])
            req_kw = np.abs(system_baseline_const[w] - target_const[w])
            tiny = 1e-6
            mae_mw[iy, ix] = float(np.mean(err_kw)) / 1000.0  # MW
            mape_pct[iy, ix] = float(np.mean(err_kw / (req_kw + tiny))) * 100.0
            cost[iy, ix] = float(total_cost)

    # Normalize cost to 0..1 for the heatmap
    cmin, cmax = float(np.min(cost)), float(np.max(cost))
    denom = (cmax - cmin) if cmax > cmin else 1.0
    cost_norm = (cost - cmin) / denom

    # Render (black out cells where MAPE > 5%)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    mask_black = mape_pct > 5.0
    render_heatmaps(COUNT_GRID, mae_mw, cost_norm, mask_black, out_dir)


if __name__ == "__main__":
    main()
