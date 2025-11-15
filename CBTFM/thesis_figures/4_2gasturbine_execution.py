"""
Figure 4.2 — Gas Turbine Execution (Single Event)

Plots three rows:
 1) Original (baseline) vs New (achieved) load
 2) Stacked contributions by agent type within the event window
 3) Error between desired reduction curve and executed reduction

Event window: 14:00–23:00 inclusive
Desired reduction (kW) provided explicitly per hour (0..23)
"""

from __future__ import annotations

import os
import sys
import random
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure interactive figures open in the system browser
pio.renderers.default = os.environ.get("PLOTLY_RENDERER", "browser")


# -----------------------------
# Configuration
# -----------------------------
USE_HEURISTIC = True

# Agent counts
N_HVAC = 10000
N_STORAGE = 6000
N_CI = 2000

N_TIMESTEPS = 24

# Event window (inclusive hour indices)
START = 14
END = 23

# Reproducibility
SEED = 20251102

# Outdoor temperature bias
OUTDOOR_TEMP_SHIFT_C = -20.0


if USE_HEURISTIC:
    from faa_agents_heuristic import (
        BuildingHVACAgentHeuristic as BuildingHVACAgent,
        EnergyStorageAgentHeuristicFast as EnergyStorageAgent,
        CILoadShiftAgentHeuristicFast as CILoadShiftAgent,
    )
else:
    from faa_agents import BuildingHVACAgent, EnergyStorageAgent, CILoadShiftAgent

from market_operator import MarketOperator


def _rng_reset(seed: int = SEED) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def build_env(T: int) -> dict:
    # Start from a smooth diurnal price curve
    base_prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    prices = base_prices.copy()

    # Make afternoon hours cheaper to encourage storage charging in the baseline.
    # For T=24 this roughly corresponds to 12:00–19:00.
    afternoon_start = int(0.5 * T)
    afternoon_end = int(0.8 * T)
    afternoon_start = max(0, min(T - 1, afternoon_start))
    afternoon_end = max(afternoon_start, min(T - 1, afternoon_end))
    prices[afternoon_start : afternoon_end + 1] *= 0.6

    outdoor = 10 + 10 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 1.6)
    outdoor = outdoor + float(OUTDOOR_TEMP_SHIFT_C)
    return {
        "outdoor_temp_c": outdoor,
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }


def build_fleet(T: int, rng: random.Random) -> List:
    agents: List = []
    # HVAC
    for i in range(N_HVAC):
        agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=float(rng.uniform(5, 15)),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=float(rng.uniform(0.4, 1.0)),
            )
        )
    # Storage
    for i in range(N_STORAGE):
        kwargs = dict(
            capacity_kwh=float(rng.uniform(10, 50)),
            max_power_kw=float(rng.uniform(2, 10)),
            cycle_cost_dol_per_kwh=float(rng.uniform(0.02, 0.08)),
        )
        if USE_HEURISTIC:
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs, lookups=None))
        else:
            agents.append(EnergyStorageAgent(f"bess_{i}", **kwargs))
    # C&I
    for i in range(N_CI):
        duration = int(rng.randint(2, 6))
        base_start = int(rng.randint(0, max(0, T - duration)))
        lo = max(0, base_start - 2)
        hi = min(T - 1, base_start + duration - 1 + 2)
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng.uniform(50, 150)),
                duration_hours=duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=float(rng.uniform(75, 200)),
            )
        )
    return agents


def collect_baseline_bids_and_schedules(
    agents: List, sim_state: dict
) -> Tuple[np.ndarray, List[Dict], Dict[str, np.ndarray]]:
    T = len(sim_state["price_dol_per_kwh"])
    baseline = np.zeros(T, dtype=float)
    all_bids: List[Dict] = []
    schedules: Dict[str, np.ndarray] = {}
    for ag in agents:
        sched = ag.get_baseline_operation(sim_state)
        schedules[getattr(ag, "agent_id")] = np.asarray(sched, dtype=float)
        baseline += sched
        # Keep per-hour bids for non-HVAC/non-Storage (avoid crowding out blocks)
        if getattr(ag, "agent_type", "") not in ("Storage", "HVAC"):
            all_bids.extend(ag.get_flexibility_bids(sim_state))
    return baseline, all_bids, schedules


def _storage_soc_trace(agent, schedule: np.ndarray) -> np.ndarray:
    C = float(getattr(agent, "capacity_kwh", 0.0))
    eff = (
        float(getattr(agent, "efficiency", 1.0))
        if getattr(agent, "efficiency", 1.0) > 0
        else 1.0
    )
    s = float(getattr(agent, "soc", 0.5)) * C
    T = len(schedule)
    trace = np.zeros(T + 1, dtype=float)
    trace[0] = s
    for t in range(T):
        p = float(schedule[t])
        if p >= 0:
            s = min(C, s + p * eff)
        else:
            s = max(0.0, s + p / eff)
        trace[t + 1] = s
    return trace


def _schedule_profit(
    prices: np.ndarray, schedule: np.ndarray, cycle_cost: float
) -> float:
    return float(np.sum(-prices * schedule - cycle_cost * np.abs(schedule)))


def synthesize_storage_window_bids(
    agents: List,
    schedules_base: Dict[str, np.ndarray],
    sim_state: dict,
    start: int,
    end: int,
) -> List[Dict]:
    prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
    T = len(prices)
    s0, e0 = max(0, int(start)), min(T - 1, int(end))
    if e0 < s0:
        s0, e0 = e0, s0

    bids: List[Dict] = []
    for ag in agents:
        if getattr(ag, "agent_type", "") != "Storage":
            continue
        aid = getattr(ag, "agent_id")
        base = np.asarray(schedules_base.get(aid, None), dtype=float)
        if base.size != T:
            continue
        P = float(getattr(ag, "max_power_kw", 0.0))
        C = float(getattr(ag, "capacity_kwh", 0.0))
        eff = (
            float(getattr(ag, "efficiency", 1.0))
            if getattr(ag, "efficiency", 1.0) > 0
            else 1.0
        )
        cc = float(getattr(ag, "cycle_cost", 0.0))
        if P <= 0 or C <= 0:
            continue

        # SOC at event start
        soc_trace = _storage_soc_trace(ag, base)
        s = float(soc_trace[s0])
        new_sched = base.copy()

        # 1) Pre-charge on cheapest pre-window hours if needed
        pre_hours = list(range(0, s0))
        if pre_hours:
            hours_window = e0 - s0 + 1
            energy_needed = max(0.0, hours_window * (P / max(eff, 1e-9)) - s)
            if energy_needed > 1e-9:
                for t in sorted(pre_hours, key=lambda t: prices[t]):
                    if energy_needed <= 1e-9:
                        break
                    headroom = max(0.0, C - s)
                    if headroom <= 1e-9:
                        break
                    p_charge = min(
                        P, headroom / max(eff, 1e-9), energy_needed / max(eff, 1e-9)
                    )
                    p = max(base[t], p_charge)
                    if p <= base[t] + 1e-12:
                        continue
                    new_sched[t] = p
                    s = min(C, s + p * eff)
                    energy_needed = max(0.0, hours_window * (P / max(eff, 1e-9)) - s)

        # 2) Discharge across the window
        for t in range(s0, e0 + 1):
            max_feasible = s * eff
            p = -min(P, max_feasible)
            if p > base[t]:  # baseline already discharging more
                p = base[t]
            new_sched[t] = p
            if p >= 0:
                s = min(C, s + p * eff)
            else:
                s = max(0.0, s + p / eff)

        # 3) Optional post-event optimization via heuristic DP if available
        if e0 + 1 < T and hasattr(ag, "_optimize_from"):
            try:
                rem_value, rem_p, rem_soc = ag._optimize_from(
                    np.asarray(sim_state["price_dol_per_kwh"], dtype=float),
                    start_t=e0 + 1,
                    start_soc_kwh=float(s),
                )
                new_sched[e0 + 1 :] = rem_p
            except Exception:
                pass

        if not np.any(np.abs(new_sched - base) > 1e-9):
            continue

        profit_base = _schedule_profit(prices, base, cc)
        profit_new = _schedule_profit(prices, new_sched, cc)
        offer_cost = max(0.0, float(profit_base - profit_new))

        bids.append(
            {
                "agent_id": aid,
                "agent_type": "Storage",
                "cost": offer_cost,
                "delta_p": new_sched - base,
            }
        )

    return bids


def _simulate_hvac_with_forced_off(
    agent, sim_state: dict, start: int, end: int
) -> np.ndarray:
    """Simulate HVAC rule but force OFF during [start, end] inclusive."""
    outdoor = np.asarray(sim_state["outdoor_temp_c"], dtype=float)
    T = len(outdoor)
    sched = np.zeros(T, dtype=float)
    temp = float(getattr(agent, "temp_c", 21.0))
    setpoint = float(sim_state["hvac_setpoint_c"])
    deadband = float(getattr(agent, "deadband_c", 0.5))
    r_val = float(getattr(agent, "r_val", 2.0))
    c_val = float(getattr(agent, "c_val", 10.0))
    p_kw = float(getattr(agent, "power_kw", 0.0))

    s0 = max(0, int(start))
    e0 = min(T - 1, int(end))
    if e0 < s0:
        s0, e0 = e0, s0

    is_heating = False
    for t in range(T):
        if s0 <= t <= e0:
            p = 0.0
            is_heating = False
        else:
            if is_heating:
                p = p_kw
            elif temp < setpoint - deadband:
                p = p_kw
                is_heating = True
            else:
                p = 0.0
            if is_heating and temp >= setpoint:
                is_heating = False
        sched[t] = p
        delta_t_out = (outdoor[t] - temp) / (r_val * c_val)
        delta_t_hvac = p / c_val
        temp = temp + delta_t_out + delta_t_hvac
    return sched


def synthesize_hvac_window_bids(
    agents: List,
    schedules_base: Dict[str, np.ndarray],
    sim_state: dict,
    start: int,
    end: int,
) -> List[Dict]:
    bids: List[Dict] = []
    T = len(sim_state["price_dol_per_kwh"])  # horizon length
    s0 = max(0, int(start))
    e0 = min(T - 1, int(end))
    if e0 < s0:
        s0, e0 = e0, s0

    for ag in agents:
        if getattr(ag, "agent_type", "") != "HVAC":
            continue
        aid = getattr(ag, "agent_id")
        base = np.asarray(schedules_base.get(aid, None), dtype=float)
        if base.size != T:
            continue
        alt = _simulate_hvac_with_forced_off(ag, sim_state, start, end)
        if not np.any(np.abs(alt - base) > 1e-9):
            continue
        try:
            base_val, _ = ag._get_path_value(base, sim_state)
            alt_val, _ = ag._get_path_value(alt, sim_state)
            cost = base_val - alt_val
        except Exception:
            prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
            cost = float(np.sum(prices * (alt - base)))

        offer_cost = max(0.0, float(cost))
        bids.append(
            {
                "agent_id": aid,
                "agent_type": "HVAC",
                "cost": offer_cost,
                "delta_p": alt - base,
            }
        )

    return bids


def build_desired_reduction_series() -> np.ndarray:
    """Desired reduction (kW) per hour for 0..23.

    The values below are provided in MW as the flexibility
    target profile; we convert them to kW for use in the
    system model.

    Provided values (interpreted as reductions, not absolute load):
    00:00–13:00 => 0.0 kW
    14:00 => 43.759...
    15:00 => 116.641...
    ...
    23:00 => 96.2658...
    """
    vals = [
        0.0,  # 00
        0.0,  # 01
        0.0,  # 02
        0.0,  # 03
        0.0,  # 04
        0.0,  # 05
        0.0,  # 06
        0.0,  # 07
        0.0,  # 08
        0.0,  # 09
        0.0,  # 10
        0.0,  # 11
        0.0,  # 12
        0.0,  # 13
        43.759380349929685,  # 14
        116.64147494901,  # 15
        141.96221860773,  # 16
        146.67419610503,  # 17
        141.63090769054,  # 18
        130.54206347448667,  # 19
        126.47585103872908,  # 20
        130.10865021739002,  # 21
        114.80085873905,  # 22
        96.26585350664901,  # 23
    ]
    arr = np.asarray(vals, dtype=float) * 1000
    if arr.size != 24:
        raise ValueError(f"Expected 24 values for desired reduction, got {arr.size}")
    return arr


def compute_target_from_reduction(
    baseline: np.ndarray, desired_reduction: np.ndarray
) -> np.ndarray:
    """Absolute target load shape = baseline - desired_reduction (clamped >= 0)."""
    tgt = np.asarray(baseline, dtype=float) - np.asarray(desired_reduction, dtype=float)
    tgt = np.maximum(0.0, tgt)
    return tgt


def render_figure(
    baseline: np.ndarray,
    target: np.ndarray,
    achieved: np.ndarray,
    dispatched: List[Dict],
    s: int,
    e: int,
):
    T = len(baseline)
    x = list(range(T))

    baseline_arr = np.asarray(baseline, dtype=float)
    achieved_arr = np.asarray(achieved, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    # Load error series in percent, measured relative to the
    # requested reduction (baseline - target). A value of:
    #   0%  => achieved == target (perfect tracking)
    # 100% => achieved == baseline (no reduction delivered)
    tiny = 1e-6
    requested_reduction = baseline_arr - target_arr
    diff = achieved_arr - target_arr
    err_pct = np.zeros_like(diff)
    valid = requested_reduction > tiny
    err_pct[valid] = 100.0 * diff[valid] / requested_reduction[valid]
    err = err_pct

    # Build figure: 3 rows
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
    )

    # Row 1 — Baseline vs Achieved vs Target
    fig.add_trace(
        go.Scatter(
            x=x,
            y=baseline,
            name="Original Load",
            mode="markers",
            marker=dict(color="#7f7f7f", size=4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=achieved,
            name="New Load",
            line=dict(color="black", width=3),
        ),
        row=1,
        col=1,
    )
    # Show target load (baseline minus flex target) in gold, within event window
    target_mask = [None] * T
    s0_top, e0_top = max(0, s), min(T - 1, e)
    for i in range(s0_top, e0_top + 1):
        target_mask[i] = float(target_arr[i])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=target_mask,
            name="Target Load",
            line=dict(color="#DAA520", width=3),
        ),
        row=1,
        col=1,
    )

    # Row 2 — Contributions stacked bars (window only)
    if dispatched:
        from collections import defaultdict

        by_type = defaultdict(list)
        for b in dispatched:
            by_type[b.get("agent_type", "Other")].append(
                np.asarray(b.get("delta_p", []), dtype=float)
            )

        colors = {
            "HVAC": "#FFA07A",
            "Storage": "#20B2AA",
            "Deferrable": "#9370DB",
            "C&I": "#4682B4",
            "Other": "#999999",
        }
        order = ["C&I", "Deferrable", "HVAC", "Storage", "Other"]

        s0, e0 = max(0, s), min(T - 1, e)
        for agent_type in order:
            if agent_type not in by_type:
                continue
            deltas = by_type[agent_type]
            total_delta = np.sum(np.stack(deltas), axis=0)
            reduction = -np.minimum(0.0, total_delta)  # positive reductions in kW
            y = [None] * T
            for i in range(s0, e0 + 1):
                y[i] = float(reduction[i])
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    name=f"{agent_type} Contribution",
                    marker_color=colors.get(agent_type),
                    showlegend=(agent_type == "C&I"),
                ),
                row=2,
                col=1,
            )

    # Row 3 — Error between target and achieved load (window only, percent)
    y_err = [None] * T
    s0, e0 = max(0, s), min(T - 1, e)
    for i in range(s0, e0 + 1):
        y_err[i] = float(err[i])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_err,
            name="Load Error (% of Target)",
            line=dict(color="#B22222", width=2),
        ),
        row=3,
        col=1,
    )

    # Shade event window across all rows
    x0 = s - 0.5
    x1 = e + 0.5
    for r in (1, 2, 3):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="rgba(255,165,0,0.12)", line_width=0, row=r, col=1
        )

    # Axes and layout
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
        margin=dict(t=50),
    )
    fig.update_yaxes(title_text="System Load (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Contrib. (kW)", row=2, col=1, nticks=3)
    fig.update_yaxes(
        title_text="Error (%)", row=3, col=1, nticks=5, showline=True, gridwidth=1
    )
    fig.update_xaxes(title_text="Hour", row=3, col=1)

    fig.show()


def main():
    rng = _rng_reset(SEED)
    sim = build_env(N_TIMESTEPS)
    agents = build_fleet(N_TIMESTEPS, rng)

    # Attach shared price lookups for heuristic storage
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

    baseline, all_bids, schedules_base = collect_baseline_bids_and_schedules(
        agents, sim
    )

    # Add multi-hour block bids for Storage and HVAC around the event window
    storage_block_bids = synthesize_storage_window_bids(
        agents, schedules_base, sim, start=START, end=END
    )
    all_bids.extend(storage_block_bids)
    hvac_block_bids = synthesize_hvac_window_bids(
        agents, schedules_base, sim, start=START, end=END
    )
    all_bids.extend(hvac_block_bids)

    desired_reduction = build_desired_reduction_series()
    target = compute_target_from_reduction(baseline, desired_reduction)

    mo = MarketOperator(peak_guard_factor=1.05)
    achieved, dispatched, total_cost = mo.clear_market(
        all_bids=all_bids,
        system_baseline_load=baseline,
        target_load_shape=target,
        focus_windows=[(START, END)],
        max_bids_per_agent=1,
    )
    print(f"Dispatch summary: {len(dispatched)} bids | total cost ${total_cost:.2f}")

    render_figure(
        baseline=baseline,
        target=target,
        achieved=achieved,
        dispatched=dispatched,
        s=START,
        e=END,
    )


if __name__ == "__main__":
    main()
