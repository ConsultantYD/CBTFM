import numpy as np
import random
import sys
import os
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faa_agents import (
    BuildingHVACAgent,
    EnergyStorageAgent,
    CILoadShiftAgent,
)
from visualizations import plot_market_dispatch_results
from market_operator import MarketOperator


def make_env(T: int) -> dict:
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))
    prices[int(0.7 * T) : min(T, int(0.7 * T) + 3)] *= 1.5
    return {
        "outdoor_temp_c": 10
        + 10 * np.sin(np.linspace(0, 2 * np.pi, T) - np.pi / 1.6),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }


def build_fleet(T: int, n_per_type: int, rng: random.Random) -> list:
    agents = []
    for i in range(n_per_type):
        agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=float(rng.uniform(5, 15)),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=float(rng.uniform(0.4, 1.0)),
            )
        )
        agents.append(
            EnergyStorageAgent(
                f"bess_{i}",
                capacity_kwh=float(rng.uniform(10, 50)),
                max_power_kw=float(rng.uniform(2, 10)),
                cycle_cost_dol_per_kwh=float(rng.uniform(0.02, 0.08)),
            )
        )
        # C&I with randomized duration and preferred window to avoid stacking
        ci_duration = int(rng.randint(2, 6))
        lo = int(rng.randint(0, max(0, T - ci_duration)))
        hi_min = lo + ci_duration - 1
        hi_max = T - 1
        hi = int(rng.randint(hi_min, min(hi_min + 8, hi_max)))
        agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=float(rng.uniform(50, 150)),
                duration_hours=ci_duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=float(rng.uniform(75, 200)),
            )
        )
    return agents


def collect_bids(agents: list, sim_state: dict):
    T = len(sim_state["price_dol_per_kwh"])
    baseline = np.zeros(T, dtype=float)
    bids = []
    for ag in agents:
        baseline += ag.get_baseline_operation(sim_state)
        bids.extend(ag.get_flexibility_bids(sim_state))
    return baseline, bids


def target_from_window_fraction(
    baseline: np.ndarray, start: int, end: int, frac_reduction: float
) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(start))
    e = min(len(baseline) - 1, int(end))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = baseline[window] * (1.0 - frac_reduction)
    return tgt


def target_from_window_cap(
    baseline: np.ndarray, start: int, end: int, absolute_cap_kw: float
) -> np.ndarray:
    tgt = baseline.copy()
    s = max(0, int(start))
    e = min(len(baseline) - 1, int(end))
    if e < s:
        s, e = e, s
    window = slice(s, e + 1)
    tgt[window] = np.minimum(baseline[window], absolute_cap_kw)
    return tgt


def main():
    print("--- Example 4: Multiple Target Shapes with the SAME Bids ---")
    T = 24
    N_PER = 25  # pick 25 per type to keep runtime snappy for multiple shapes
    rng = random.Random(42)

    # 1) Environment + Agents + Bids (done once)
    print("1. Building environment and collecting a single set of bids...")
    sim_state = make_env(T)
    agents = build_fleet(T, N_PER, rng)
    baseline, all_bids = collect_bids(agents, sim_state)
    print(f"Collected {len(all_bids)} bids from {len(agents)} agents.")

    # 2) Define several event windows and corresponding target shapes
    print("2. Defining multiple event windows and their target shapes...")
    events: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
    # Event A: 2-hour full shed (target = 0)
    tgt_A = target_from_window_cap(baseline, 7, 8, 0.0)
    events.append(("2h full shed to 0 (07–08)", (7, 8), tgt_A))
    # Event B: 2-hour absolute cap at 0 during midday (demonstrates single-hour → 2-hour change)
    sB, eB = 13, 14
    tgt_B = target_from_window_cap(baseline, sB, eB, 0.0)
    events.append(("2h full shed to 0 (13–14)", (sB, eB), tgt_B))
    # Event C: 4-hour 25% cap during evening
    tgt_C = target_from_window_fraction(baseline, 18, 21, 0.25)
    events.append(("4h -25% reduction (18–21)", (18, 21), tgt_C))

    # 3) Clear each event with the SAME bids, focusing the MO objective only on the event window
    mo = MarketOperator(peak_guard_factor=1.05)
    for name, (s, e), target in events:
        print(f"\nClearing event: {name}")
        achieved, dispatched, total_cost = mo.clear_market(
            all_bids=all_bids,
            system_baseline_load=baseline,
            target_load_shape=target,
            focus_windows=[(s, e)],
        )
        print(
            f"Peak (window): baseline {np.max(baseline[s:e+1]):.2f} → target {np.max(target[s:e+1]):.2f} → achieved {np.max(achieved[s:e+1]):.2f} kW"
        )
        print(f"Dispatched {len(dispatched)} bids | Total cost ${total_cost:.2f}")
        plot_market_dispatch_results(
            baseline,
            target,
            achieved,
            dispatched,
            window_start=s,
            window_end=e,
        )


if __name__ == "__main__":
    main()
