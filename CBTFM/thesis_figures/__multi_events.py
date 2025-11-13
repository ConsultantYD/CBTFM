"""
Figure 2 — Example 4 (three events) combined into a single panel.

Build a small fleet, collect one set of bids, then clear three different
target shapes (events) with the SAME bids. Display results in one figure
with 3 rows x 1 column. Each row shows:
 - System Baseline (dashed)
 - Load Target (red, visible only inside the event window)
 - Achieved Load (black)
 - Stacked bars by agent type (on secondary y-axis), shown only within the window

Debug speed: 5 agents per type. You can increase later for publication.
"""

from __future__ import annotations

import os
import sys
import random
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Toggle heuristic vs exact agent implementations
USE_HEURISTIC = False

# Ensure interactive figures open in the system browser
pio.renderers.default = os.environ.get("PLOTLY_RENDERER", "browser")

if USE_HEURISTIC:
    from faa_agents_heuristic import (
        BuildingHVACAgentHeuristic as BuildingHVACAgent,
        EnergyStorageAgentHeuristicFast as EnergyStorageAgent,
        CILoadShiftAgentHeuristicFast as CILoadShiftAgent,
    )
else:
    from faa_agents import BuildingHVACAgent, EnergyStorageAgent, CILoadShiftAgent
from market_operator import MarketOperator


# Reproducibility
SEED = 20251102
N_TIMESTEPS = 24
N_PER_TYPE = 50  # debug fast; increase later for publication


def build_env(T: int) -> dict:
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    prices[int(0.7 * T) : min(T, int(0.7 * T) + 3)] *= 1.5
    return {
        "outdoor_temp_c": 10 + 10 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 1.6),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }


def build_fleet(T: int, n: int, rng: random.Random) -> list:
    agents = []
    # If using heuristic storage, precompute price lookups once for all ESS
    price_lookup = None
    if USE_HEURISTIC:
        # We'll compute lookups later once we have prices in env, so keep None here
        pass
    for i in range(n):
        agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=float(rng.uniform(5, 15)),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=float(rng.uniform(0.4, 1.0)),
            )
        )
        # Storage (pass lookups if using heuristic fast variant)
        kwargs = dict(
            capacity_kwh=float(rng.uniform(10, 50)),
            max_power_kw=float(rng.uniform(2, 10)),
            cycle_cost_dol_per_kwh=float(rng.uniform(0.02, 0.08)),
        )
        if USE_HEURISTIC:
            # defer lookups assignment in main where prices are known
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs, lookups=None))
        else:
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs))
        # C&I with randomized window to avoid stacking
        dur = int(rng.randint(2, 6))
        lo = int(rng.randint(0, max(0, T - dur)))
        hi_min, hi_max = lo + dur - 1, T - 1
        hi = int(rng.randint(hi_min, min(hi_min + 8, hi_max)))
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng.uniform(50, 150)),
                duration_hours=dur,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=float(rng.uniform(75, 200)),
            )
        )
    return agents


def collect_baseline_and_bids(agents: list, sim_state: dict) -> tuple[np.ndarray, list]:
    T = len(sim_state["price_dol_per_kwh"])
    baseline = np.zeros(T, dtype=float)
    bids = []
    for ag in agents:
        baseline += ag.get_baseline_operation(sim_state)
        bids.extend(ag.get_flexibility_bids(sim_state))
    return baseline, bids


def target_from_window_fraction(
    baseline: np.ndarray, s: int, e: int, frac: float
) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = baseline[window] * (1.0 - frac)
    return tgt


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


def add_event_section(
    fig,
    top_row: int,
    bottom_row: int,
    baseline: np.ndarray,
    target: np.ndarray,
    achieved: np.ndarray,
    dispatched_bids: List[dict],
    s: int,
    e: int,
    *,
    show_legend: bool = False,
    show_bar_legend: bool = False,
) -> None:
    """Add two rows for one event: top=lines, bottom=bars (within window only)."""
    T = len(baseline)
    x = list(range(T))

    # Top row: baseline (dashed grey), achieved (black), target (gold masked to window)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=baseline,
            name="System Baseline",
            line=dict(color="#7f7f7f", dash="dash"),
            showlegend=show_legend,
        ),
        row=top_row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=achieved,
            name="Achieved Load",
            line=dict(color="black", width=3),
            showlegend=show_legend,
        ),
        row=top_row,
        col=1,
    )

    masked_target = [None] * T
    s0 = max(0, s)
    e0 = min(T - 1, e)
    for i in range(s0, e0 + 1):
        masked_target[i] = float(target[i])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=masked_target,
            name="Load Target",
            line=dict(color="#DAA520", width=3),
            showlegend=show_legend,
        ),
        row=top_row,
        col=1,
    )

    # Bottom row: stacked bars by agent type inside window only (legend optional)
    if dispatched_bids:
        from collections import defaultdict

        by_type = defaultdict(list)
        for b in dispatched_bids:
            by_type[b.get("agent_type", "Other")].append(
                np.asarray(b.get("delta_p", []), dtype=float)
            )
        # Consistent colors
        colors = {
            "HVAC": "#FFA07A",
            "Storage": "#20B2AA",
            "Deferrable": "#9370DB",
            "C&I": "#4682B4",  # always plotted first (bottom of stack)
            "Other": "#999999",
        }
        # Plot in a fixed order so C&I is bottom in the stack
        order = ["C&I", "Deferrable", "HVAC", "Storage", "Other"]
        for agent_type in order:
            if agent_type not in by_type:
                continue
            deltas = by_type[agent_type]
            total_delta = np.sum(np.stack(deltas), axis=0)
            reduction = -np.minimum(0.0, total_delta)  # positive reductions
            y = [None] * T
            for i in range(s0, e0 + 1):
                y[i] = float(reduction[i])
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    name=f"{agent_type} Contribution",
                    marker_color=colors.get(agent_type),
                    showlegend=show_bar_legend,
                ),
                row=bottom_row,
                col=1,
            )

    # Shade event window on both rows
    x0 = s - 0.5
    x1 = e + 0.5
    for r in (top_row, bottom_row):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(255,165,0,0.12)",
            line_width=0,
            row=r,
            col=1,
        )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    sim = build_env(N_TIMESTEPS)
    fleet = build_fleet(N_TIMESTEPS, N_PER_TYPE, random)
    # If heuristic storage is used, populate lookups now that prices are known
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
    baseline, all_bids = collect_baseline_and_bids(fleet, sim)

    # Define three events (same spirit as Example 4)
    events: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
    tgt_A = target_from_window_cap(baseline, 7, 8, 0.0)
    events.append(("2h full shed to 0 (07–08)", (7, 8), tgt_A))
    tgt_B = target_from_window_cap(baseline, 13, 14, 0.0)
    events.append(("2h full shed to 0 (13–14)", (13, 14), tgt_B))
    tgt_C = target_from_window_fraction(baseline, 18, 21, 0.25)
    events.append(("4h -25% reduction (18–21)", (18, 21), tgt_C))

    mo = MarketOperator(peak_guard_factor=1.05)

    # Build a single figure with 4 rows (2 per event)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.28, 0.12, 0.28, 0.12],
    )

    # Keep two case studies (drop the middle one)
    events_to_plot = [events[0], events[2]]
    for idx, (name, (s, e), tgt) in enumerate(events_to_plot, start=1):
        achieved, dispatched, total_cost = mo.clear_market(
            all_bids=all_bids,
            system_baseline_load=baseline,
            target_load_shape=tgt,
            focus_windows=[(s, e)],
        )
        top = 2 * (idx - 1) + 1
        bottom = top + 1
        add_event_section(
            fig,
            top,
            bottom,
            baseline,
            tgt,
            achieved,
            dispatched,
            s,
            e,
            show_legend=(idx == 1),
            show_bar_legend=(idx == 1),
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=11),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
        margin=dict(t=50),
    )
    # Y-axis titles for top rows only; limit bottom rows to ~2-3 ticks
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(nticks=3, row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=3, col=1)
    fig.update_yaxes(nticks=3, row=4, col=1)
    fig.update_xaxes(title_text="Hour", row=4, col=1)

    # Show interactively in browser (no local file save)
    fig.show()


if __name__ == "__main__":
    main()
