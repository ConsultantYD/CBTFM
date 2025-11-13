"""
Figure 5.2 — Congestion Relief (Dual Populations)

Two simultaneous populations with the same fleet size and environment as 4_2:
 - Sink: shed as much as possible (cap to 0) during 18–20.
 - Source: consume as much as possible during 18–20.

We clear markets independently for each population, then render both results in
one figure with 2 rows x 1 column:
  - Top: Source — Original vs New load (18–20)
  - Bottom: Sink — Original vs New load (18–20)

The idea is to show theoretical congestion relief: consume more at the source
while shedding at the sink over the same window.
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

N_HVAC = 1000
N_STORAGE = 50
N_CI = 50

N_TIMESTEPS = 24
START = 18  # 6 PM
END = 20  # 8 PM (inclusive)

SEED = 20251102
OUTDOOR_TEMP_SHIFT_C = -5.0


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
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    prices[int(0.7 * T) : min(T, int(0.7 * T) + 3)] *= 1.5
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


def synthesize_storage_window_bids_down(
    agents: List,
    schedules_base: Dict[str, np.ndarray],
    sim_state: dict,
    start: int,
    end: int,
) -> List[Dict]:
    """Storage block bids to reduce load (discharge) inside [start, end]."""
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

        soc_trace = _storage_soc_trace(ag, base)
        s = float(soc_trace[s0])
        new_sched = base.copy()

        # Pre-charge on cheapest pre-window hours if needed
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

        # Discharge in window
        for t in range(s0, e0 + 1):
            max_feasible = s * eff
            p = -min(P, max_feasible)
            if p > base[t]:
                p = base[t]
            new_sched[t] = p
            if p >= 0:
                s = min(C, s + p * eff)
            else:
                s = max(0.0, s + p / eff)

        if e0 + 1 < T and hasattr(ag, "_optimize_from"):
            try:
                _, rem_p, _ = ag._optimize_from(
                    prices, start_t=e0 + 1, start_soc_kwh=float(s)
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


def synthesize_storage_window_bids_up(
    agents: List,
    schedules_base: Dict[str, np.ndarray],
    sim_state: dict,
    start: int,
    end: int,
) -> List[Dict]:
    """Storage block bids to increase load (charge) inside [start, end]."""
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

        soc_trace = _storage_soc_trace(ag, base)
        s = float(soc_trace[s0])
        new_sched = base.copy()

        # Pre-discharge to free headroom: choose highest-price hours first
        pre_hours = list(range(0, s0))
        for t in sorted(pre_hours, key=lambda t: -prices[t]):
            headroom = C - s
            if headroom >= C - 1e-6:  # already empty enough
                pass
            if s <= 1e-9:
                break
            p_discharge = -min(P, s / max(eff, 1e-9))
            p = min(base[t], p_discharge)
            if p >= base[t] - 1e-12:
                continue
            new_sched[t] = p
            s = max(0.0, s + p / eff)

        # Charge in window
        for t in range(s0, e0 + 1):
            headroom = C - s
            if headroom <= 1e-9:
                p = base[t]
            else:
                p_charge = min(P, headroom / max(eff, 1e-9))
                p = max(base[t], p_charge)
            new_sched[t] = p
            if p >= 0:
                s = min(C, s + p * eff)
            else:
                s = max(0.0, s + p / eff)

        if e0 + 1 < T and hasattr(ag, "_optimize_from"):
            try:
                _, rem_p, _ = ag._optimize_from(
                    prices, start_t=e0 + 1, start_soc_kwh=float(s)
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


def _simulate_hvac_with_forced_on(
    agent, sim_state: dict, start: int, end: int
) -> np.ndarray:
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
            p = p_kw
            is_heating = True
        else:
            if is_heating and temp >= setpoint:
                is_heating = False
            if is_heating:
                p = p_kw
            elif temp < setpoint - deadband:
                p = p_kw
                is_heating = True
            else:
                p = 0.0
        sched[t] = p
        delta_t_out = (outdoor[t] - temp) / (r_val * c_val)
        delta_t_hvac = p / c_val
        temp = temp + delta_t_out + delta_t_hvac
    return sched


def synthesize_hvac_window_bids_down(
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


def synthesize_hvac_window_bids_up(
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
        alt = _simulate_hvac_with_forced_on(ag, sim_state, start, end)
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


def cap_target(baseline: np.ndarray, s: int, e: int, cap_kw: float) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    tgt[s : e + 1] = np.minimum(baseline[s : e + 1], cap_kw)
    return tgt


def boost_target(
    baseline: np.ndarray, s: int, e: int, amount_kw: float = 1e6
) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    tgt[s : e + 1] = baseline[s : e + 1] + amount_kw
    return tgt


def render_dual_figure(
    baseline_left: np.ndarray,
    achieved_left: np.ndarray,
    dispatched_left: List[Dict],
    baseline_right: np.ndarray,
    achieved_right: np.ndarray,
    dispatched_right: List[Dict],
    s: int,
    e: int,
):
    T = len(baseline_left)
    x = list(range(T))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Original vs New (stacked vertically) — Source on top, Sink bottom
    for row_idx, (baseline, achieved) in enumerate(
        [(baseline_right, achieved_right), (baseline_left, achieved_left)], start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=baseline,
                name="Original Load",
                mode="markers",
                marker=dict(color="#7f7f7f", size=4),
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=achieved,
                name="New Load",
                line=dict(color="black", width=3),
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )

    # Shade event window
    x0 = s - 0.5
    x1 = e + 0.5
    for r in (1, 2):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="rgba(255,165,0,0.12)", line_width=0, row=r, col=1
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0),
        margin=dict(t=60),
    )
    fig.update_yaxes(title_text="Source Load", row=1, col=1)
    fig.update_yaxes(title_text="Sink Load", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=1)

    fig.show()


def main():
    rng = _rng_reset(SEED)
    sim = build_env(N_TIMESTEPS)

    # Build two independent populations (sink and source)
    pop_sink = build_fleet(N_TIMESTEPS, rng)
    pop_source = build_fleet(N_TIMESTEPS, rng)

    # Attach shared price lookups for heuristic storage
    if USE_HEURISTIC:
        try:
            from faa_agents_heuristic import EnergyStorageAgentHeuristicFast as _ESS

            lookups = _ESS.precompute_lookups(sim["price_dol_per_kwh"])
            for ag in pop_sink + pop_source:
                if getattr(ag, "agent_type", "") == "Storage" and hasattr(
                    ag, "_lookups"
                ):
                    ag._lookups = lookups
        except Exception:
            pass

    # Collect baselines and per-hour bids (excluding HVAC/Storage which we synthesize as blocks)
    base_sink, bids_sink, scheds_sink = collect_baseline_bids_and_schedules(
        pop_sink, sim
    )
    base_source, bids_source, scheds_source = collect_baseline_bids_and_schedules(
        pop_source, sim
    )

    # Add block bids tailored to each direction
    bids_sink.extend(
        synthesize_storage_window_bids_down(pop_sink, scheds_sink, sim, START, END)
    )
    bids_sink.extend(
        synthesize_hvac_window_bids_down(pop_sink, scheds_sink, sim, START, END)
    )

    bids_source.extend(
        synthesize_storage_window_bids_up(pop_source, scheds_source, sim, START, END)
    )
    bids_source.extend(
        synthesize_hvac_window_bids_up(pop_source, scheds_source, sim, START, END)
    )

    # Targets
    target_sink = cap_target(base_sink, START, END, 0.0)
    # Big boost target ensures up-request selection; guard loosened for this side
    target_source = boost_target(base_source, START, END, amount_kw=1e6)

    mo_down = MarketOperator(peak_guard_factor=1.05)
    mo_up = MarketOperator(peak_guard_factor=1000.0)

    achieved_sink, dispatched_sink, cost_sink = mo_down.clear_market(
        all_bids=bids_sink,
        system_baseline_load=base_sink,
        target_load_shape=target_sink,
        focus_windows=[(START, END)],
        max_bids_per_agent=1,
    )

    achieved_source, dispatched_source, cost_source = mo_up.clear_market(
        all_bids=bids_source,
        system_baseline_load=base_source,
        target_load_shape=target_source,
        focus_windows=[(START, END)],
        max_bids_per_agent=1,
    )

    print(
        f"Sink: {len(dispatched_sink)} bids, cost ${cost_sink:.2f} | "
        f"Source: {len(dispatched_source)} bids, cost ${cost_source:.2f}"
    )

    render_dual_figure(
        baseline_left=base_sink,
        achieved_left=achieved_sink,
        dispatched_left=dispatched_sink,
        baseline_right=base_source,
        achieved_right=achieved_source,
        dispatched_right=dispatched_source,
        s=START,
        e=END,
    )


if __name__ == "__main__":
    main()
