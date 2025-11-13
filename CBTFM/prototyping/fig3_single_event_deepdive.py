"""
Figure 3 — Single Event Deep Dive (Publication Grade)

Runs one late-afternoon event (default 18–21) with a shed-to-zero target.
Top panel shows Baseline/Target/Achieved. Bottom four equally-sized panels show:
 A) Dispatched contributions by agent type (stacked bars in the event window)
 B) Average State of Charge (SOC) across energy storage assets (post-dispatch)
 C) Average indoor temperature across buildings (post-dispatch)
 D) Remaining scheduled C&I energy to be deployed (kWh, post-dispatch)

You can control the number of assets per type and the plotted time window via constants.
Heuristic agents are enabled by default via USE_HEURISTIC.
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

# -----------------------------
# Configuration constants
# -----------------------------
USE_HEURISTIC = True

# Agent counts
N_HVAC = 300
N_STORAGE = 50
N_CI = 50

# Time horizon and view window (hours are inclusive indices 0..23)
N_TIMESTEPS = 24
VIEW_START_HOUR = 0
VIEW_END_HOUR = 23

# Event window and target
START = 17
END = 19
TARGET_CAP_KW = 0.0  # shed to zero in the event window

# Reproducibility
SEED = 20251102

# C&I distribution controls (disabled to match other figures' random mix)
CI_EVEN_DISTRIBUTION = False
CI_WINDOW_SLACK_HRS = 2

# Outdoor temperature control (colder makes HVAC run more often). Negative values
# shift the outdoor series downward.
OUTDOOR_TEMP_SHIFT_C = -8.0

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
    # Storage (attach lookups later if heuristic)
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
    # C&I (evenly distribute preferred windows across the day for visibility)
    for i in range(N_CI):
        duration = int(rng.randint(2, 6))
        if CI_EVEN_DISTRIBUTION and N_CI > 1:
            base_start = int(round(i * (T - duration) / (N_CI - 1)))
        else:
            base_start = int(rng.randint(0, max(0, T - duration)))
        lo = max(0, base_start - int(CI_WINDOW_SLACK_HRS))
        hi = min(T - 1, base_start + duration - 1 + int(CI_WINDOW_SLACK_HRS))
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng.uniform(50, 150)),
                duration_hours=duration,
                preferred_window_hours=(lo, hi),
                # Use a stronger outside-window penalty (aligned with other figures)
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
        # For fig3, prefer full-path multi-hour block bids synthesized below for
        # Storage and HVAC. Exclude their per-hour bids to avoid crowding out
        # block offers under the one-bid-per-agent constraint.
        if getattr(ag, "agent_type", "") not in ("Storage", "HVAC"):
            all_bids.extend(ag.get_flexibility_bids(sim_state))
    return baseline, all_bids, schedules


def cap_target(baseline: np.ndarray, s: int, e: int, cap_kw: float) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(s))
    e = min(len(baseline) - 1, int(e))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = np.minimum(baseline[window], cap_kw)
    return tgt


def apply_dispatched_deltas(
    schedules: Dict[str, np.ndarray], dispatched: List[Dict]
) -> Dict[str, np.ndarray]:
    new_scheds = {k: v.copy() for k, v in schedules.items()}
    for b in dispatched:
        aid = b.get("agent_id")
        if aid in new_scheds:
            delta = np.asarray(b.get("delta_p", []), dtype=float)
            if delta.shape == new_scheds[aid].shape:
                new_scheds[aid] = new_scheds[aid] + delta
    return new_scheds


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
    """Synthesize multi-hour storage discharge bids with optional pre/post actions.

    - Pre-charge on the cheapest hours before the window to build SOC (if needed).
    - Discharge across the entire event window up to power/SOC limits.
    - Post-charge on the cheapest hours after the window to restore SOC to the
      event-start level (so that follow-on behavior can differ from baseline if desired).
    - Price as true schedule opportunity cost vs. baseline.
    """
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
        s_start = float(s)
        new_sched = base.copy()

        # 1) Pre-charge on cheapest pre-window hours if needed
        pre_hours = list(range(0, s0))
        if pre_hours:
            hours_window = e0 - s0 + 1
            energy_needed = max(
                0.0, hours_window * (P / max(eff, 1e-9)) - s
            )  # kWh target to sustain P
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

        # 3) Post-event profit optimization via DP on the remainder of the horizon
        #    starting from SOC at end of event. This allows recharging after the
        #    event and later discharging if profitable (even outside the window).
        if e0 + 1 < T and hasattr(ag, "_optimize_from"):
            try:
                rem_value, rem_p, rem_soc = ag._optimize_from(
                    np.asarray(sim_state["price_dol_per_kwh"], dtype=float),
                    start_t=e0 + 1,
                    start_soc_kwh=float(s),
                )
                new_sched[e0 + 1 :] = rem_p
            except Exception:
                # If DP fails for any reason, keep the baseline suffix (no change)
                pass

        # Skip if no change
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


def compute_avg_storage_soc(
    agents: List, schedules: Dict[str, np.ndarray]
) -> np.ndarray:
    socs = []
    for ag in agents:
        if getattr(ag, "agent_type", "") != "Storage":
            continue
        sched = schedules.get(getattr(ag, "agent_id"))
        if sched is None:
            continue
        T = len(sched)
        C = float(getattr(ag, "capacity_kwh", 0.0))
        eff = (
            float(getattr(ag, "efficiency", 1.0))
            if getattr(ag, "efficiency", 1.0) > 0
            else 1.0
        )
        s = float(getattr(ag, "soc", 0.5)) * C
        trace = np.zeros(T, dtype=float)
        for t in range(T):
            p = float(sched[t])
            if p >= 0:
                s = min(C, s + p * eff)
            else:
                s = max(0.0, s + p / eff)
            trace[t] = s / C if C > 0 else 0.0
        socs.append(trace)
    if not socs:
        return np.zeros(N_TIMESTEPS)
    return np.mean(np.vstack(socs), axis=0)


def compute_avg_building_temp(
    agents: List, schedules: Dict[str, np.ndarray], sim_state: dict
) -> np.ndarray:
    temps = []
    for ag in agents:
        if getattr(ag, "agent_type", "") != "HVAC":
            continue
        sched = schedules.get(getattr(ag, "agent_id"))
        if sched is None:
            continue
        # Prefer built-in sim if available
        if hasattr(ag, "_sim_temp"):
            tr = ag._sim_temp(np.asarray(sched, dtype=float), sim_state)
        else:
            # Fallback simple RC step
            r_val = float(getattr(ag, "r_val", 2.0))
            c_val = float(getattr(ag, "c_val", 10.0))
            temp = float(getattr(ag, "temp_c", 21.0))
            outdoor = np.asarray(sim_state["outdoor_temp_c"], dtype=float)
            tr_list = []
            for t, p in enumerate(sched):
                delta_t_out = (outdoor[t] - temp) / (r_val * c_val)
                delta_t_hvac = float(p) / c_val
                temp = temp + delta_t_out + delta_t_hvac
                tr_list.append(temp)
            tr = np.asarray(tr_list, dtype=float)
        temps.append(tr)
    if not temps:
        return np.zeros(N_TIMESTEPS)
    return np.mean(np.vstack(temps), axis=0)


def compute_ci_total_load(agents: List, schedules: Dict[str, np.ndarray]) -> np.ndarray:
    """Total instantaneous C&I load (kW) across all C&I agents, per time step."""
    total = np.zeros(N_TIMESTEPS, dtype=float)
    for ag in agents:
        if getattr(ag, "agent_type", "") != "C&I":
            continue
        aid = getattr(ag, "agent_id")
        sched = schedules.get(aid)
        if sched is None:
            continue
        total += np.asarray(sched, dtype=float)
    return total


def _simulate_hvac_with_forced_off(
    agent,
    sim_state: dict,
    start: int,
    end: int,
) -> np.ndarray:
    """Simulate an HVAC schedule identical to the agent's heuristic control,
    except forced OFF (p=0) during [start, end] inclusive.

    Uses the same rule as get_baseline_operation (simple threshold + latch),
    but enforces p=0 in the event window and resumes the rule outside.
    """
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
        # Decide action
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

            # Stop heating when we reach setpoint
            if is_heating and temp >= setpoint:
                is_heating = False

        sched[t] = p

        # Propagate temperature
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
    """Create full-path HVAC bids that enforce OFF during [start, end],
    letting the rule resume outside the window. This encodes post-event
    rebound (and any pre-event changes) into the bid's delta.
    """
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
        # Build alternative full schedule with forced-off window
        alt = _simulate_hvac_with_forced_off(ag, sim_state, start, end)
        if not np.any(np.abs(alt - base) > 1e-9):
            continue

        # Price using the agent's path value function (true opportunity cost)
        try:
            base_val, _ = ag._get_path_value(base, sim_state)
            alt_val, _ = ag._get_path_value(alt, sim_state)
            cost = base_val - alt_val
        except Exception:
            # Fallback: approximate by energy-only deltas
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


def render_figure(
    baseline: np.ndarray,
    target: np.ndarray,
    achieved: np.ndarray,
    dispatched: List[Dict],
    avg_soc: np.ndarray,
    avg_temp: np.ndarray,
    ci_total: np.ndarray,
    s: int,
    e: int,
    view_lo: int,
    view_hi: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    T = len(baseline)
    x = list(range(T))

    # Mask for view window
    lo = max(0, int(view_lo))
    hi = min(T - 1, int(view_hi))
    x_range = [lo - 0.5, hi + 0.5]

    # Build figure: 1 large row + 4 small rows
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.52, 0.12, 0.12, 0.12, 0.12],
    )

    # Row 1 — baseline, achieved, target
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
    fig.add_trace(
        go.Scatter(
            x=x, y=achieved, name="Achieved Load", line=dict(color="black", width=3)
        ),
        row=1,
        col=1,
    )
    mask_tgt = [None] * T
    for i in range(max(0, s), min(T - 1, e) + 1):
        mask_tgt[i] = float(target[i])
    fig.add_trace(
        go.Scatter(
            x=x, y=mask_tgt, name="Load Target", line=dict(color="#DAA520", width=3)
        ),
        row=1,
        col=1,
    )

    # Row 2 — A) Contributions stacked bars within event window
    contrib_max = 0.0
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

        # Precompute reductions per type and total stacked max in window
        reductions_by_type = {}
        stacked = np.zeros(T, dtype=float)
        s0, e0 = max(0, s), min(T - 1, e)
        for agent_type in order:
            if agent_type not in by_type:
                continue
            deltas = by_type[agent_type]
            total_delta = np.sum(np.stack(deltas), axis=0)
            reduction = -np.minimum(0.0, total_delta)
            reductions_by_type[agent_type] = reduction
            stacked += reduction
        if e0 >= s0:
            contrib_max = float(np.max(stacked[s0 : e0 + 1]))

        # Plot bars in order
        for agent_type in order:
            if agent_type not in reductions_by_type:
                continue
            reduction = reductions_by_type[agent_type]
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
    # Scale y-axis for contributions
    ymax = contrib_max if contrib_max > 0 else 1e-6
    fig.update_yaxes(range=[0, ymax], nticks=3, row=2, col=1)

    # Row 3 — B) Avg SOC (percent)
    soc_pct = avg_soc * 100.0
    fig.add_trace(
        go.Scatter(
            x=x, y=soc_pct, name="Avg Storage SOC (%)", line=dict(color="#20B2AA")
        ),
        row=3,
        col=1,
    )
    # Scale y-axis 0..max% in view window, few ticks
    soc_max = float(np.max(soc_pct[lo : hi + 1])) if hi >= lo else 0.0
    fig.update_yaxes(
        title_text="Avg SOC (%)",
        row=3,
        col=1,
        range=[0, soc_max if soc_max > 0 else 1e-6],
        nticks=3,
    )

    # Row 4 — C) Avg Temp
    fig.add_trace(
        go.Scatter(
            x=x, y=avg_temp, name="Avg Building Temp (°C)", line=dict(color="#ff7f0e")
        ),
        row=4,
        col=1,
    )
    if hi >= lo:
        temp_min = float(np.min(avg_temp[lo : hi + 1]))
        temp_max = float(np.max(avg_temp[lo : hi + 1]))
    else:
        temp_min, temp_max = 0.0, 1.0
    if abs(temp_max - temp_min) < 1e-9:
        temp_max = temp_min + 1e-6
    fig.update_yaxes(
        title_text="Temp (°C)",
        row=4,
        col=1,
        range=[temp_min, temp_max],
        nticks=3,
    )

    # Row 5 — D) Total C&I live load (kW) as a simple line
    fig.add_trace(
        go.Scatter(
            x=x, y=ci_total, name="Total C&I Load (kW)", line=dict(color="#4682B4")
        ),
        row=5,
        col=1,
    )
    ci_max = float(np.max(ci_total[lo : hi + 1])) if hi >= lo else 0.0
    fig.update_yaxes(
        title_text="C&I (kW)",
        row=5,
        col=1,
        range=[0, ci_max if ci_max > 0 else 1e-6],
        nticks=3,
    )

    # Shade event window across all rows
    x0 = s - 0.5
    x1 = e + 0.5
    for r in range(1, 6):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="rgba(255,165,0,0.12)", line_width=0, row=r, col=1
        )

    # Styling
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0.0),
        margin=dict(t=50),
        xaxis=dict(range=x_range),
        xaxis2=dict(range=x_range),
        xaxis3=dict(range=x_range),
        xaxis4=dict(range=x_range),
        xaxis5=dict(range=x_range),
    )
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=5, col=1)

    # Smaller fonts for bottom panels to reduce label overlap
    small_title = dict(size=10)
    small_ticks = dict(size=9)
    for r in (2, 3, 4, 5):
        fig.update_yaxes(row=r, col=1, title_font=small_title, tickfont=small_ticks)

    # Save outputs
    pdf_path = os.path.abspath(
        os.path.join(out_dir, "figure_03_single_event_deepdive.pdf")
    )
    png_path = os.path.abspath(
        os.path.join(out_dir, "figure_03_single_event_deepdive.png")
    )
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
    target = cap_target(baseline, START, END, TARGET_CAP_KW)

    mo = MarketOperator(peak_guard_factor=1.05)

    # Enable full-path multi-hour block bids (pre-charge/rebound) in addition to per-hour bids
    storage_block_bids = synthesize_storage_window_bids(
        agents, schedules_base, sim, start=START, end=END
    )
    all_bids.extend(storage_block_bids)
    hvac_block_bids = synthesize_hvac_window_bids(
        agents, schedules_base, sim, start=START, end=END
    )
    all_bids.extend(hvac_block_bids)
    achieved, dispatched, total_cost = mo.clear_market(
        all_bids=all_bids,
        system_baseline_load=baseline,
        target_load_shape=target,
        focus_windows=[(START, END)],
        max_bids_per_agent=1,
    )
    print(f"Dispatch summary: {len(dispatched)} bids | total cost ${total_cost:.2f}")

    # Apply dispatched deltas to per-agent schedules to derive post-dispatch schedules
    schedules_post = apply_dispatched_deltas(schedules_base, dispatched)
    avg_soc = compute_avg_storage_soc(agents, schedules_post)
    avg_temp = compute_avg_building_temp(agents, schedules_post, sim)
    ci_total = compute_ci_total_load(agents, schedules_post)

    # Save under project-level results/
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    render_figure(
        baseline,
        target,
        achieved,
        dispatched,
        avg_soc,
        avg_temp,
        ci_total,
        s=START,
        e=END,
        view_lo=VIEW_START_HOUR,
        view_hi=VIEW_END_HOUR,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
