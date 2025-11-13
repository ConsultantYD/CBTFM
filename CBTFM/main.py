import numpy as np
import random
from typing import List, Dict

from faa_agents import (
    BuildingHVACAgent,
    EnergyStorageAgent,
    DeferrableLoadAgent,
    CILoadShiftAgent,
)
from visualizations import (
    plot_flexibility_supply_curve,
    plot_market_dispatch_results,
    plot_cost_percentile_stack,
)


def run_market_clearing(
    baseline_load: np.ndarray, target_load: np.ndarray, all_bids: List[Dict]
):
    """
    Greedy algorithm that respects the "One Bid Per Agent" rule, dispatching bids
    based on their cost-effectiveness at reducing error against the target shape.
    """
    achieved_load = baseline_load.copy()
    dispatched_bids = []
    total_cost = 0.0

    available_bids = {}
    for bid in all_bids:
        if bid["cost"] >= 0:
            if bid["agent_id"] not in available_bids:
                available_bids[bid["agent_id"]] = []
            available_bids[bid["agent_id"]].append(bid)

    while True:
        best_bid_info = {"bid": None, "score": float("inf")}
        current_error = np.sum((achieved_load - target_load) ** 2)

        for agent_id, bids in available_bids.items():
            for bid in bids:
                proposed_load = achieved_load + bid["delta_p"]

                # Heuristic to prevent creating new, worse peaks
                if np.max(proposed_load) > np.max(achieved_load) * 1.05:
                    continue

                new_error = np.sum((proposed_load - target_load) ** 2)
                error_reduction = current_error - new_error

                if error_reduction > 1e-6:
                    score = (
                        bid["cost"] / error_reduction
                    )  # Cost per unit of squared error reduction
                    if score < best_bid_info["score"]:
                        best_bid_info = {
                            "bid": bid,
                            "score": score,
                            "new_load": proposed_load,
                        }

        if best_bid_info["bid"] is not None:
            best_bid = best_bid_info["bid"]
            achieved_load = best_bid_info["new_load"]
            total_cost += best_bid["cost"]
            dispatched_bids.append(best_bid)
            del available_bids[best_bid["agent_id"]]
        else:
            break

    return achieved_load, dispatched_bids, total_cost


def main():
    print("--- CBTFM Simulation Start ---")
    N_TIMESTEPS = 24
    N_AGENTS_PER_TYPE = 25

    print("1. Creating simulation environment...")
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, N_TIMESTEPS)))
    prices[17:20] *= 1.5
    prices += 0.02 * np.random.rand(N_TIMESTEPS)
    sim_state = {
        "outdoor_temp_c": 10
        + 10 * np.sin(np.linspace(0, 2 * np.pi, N_TIMESTEPS) - np.pi / 1.5),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }

    print(f"2. Instantiating {N_AGENTS_PER_TYPE * 4} total agents...")
    all_agents = []
    for i in range(N_AGENTS_PER_TYPE):
        all_agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=random.uniform(5, 15),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=random.uniform(0.4, 1.0),
            )
        )
        all_agents.append(
            EnergyStorageAgent(
                f"bess_{i}",
                capacity_kwh=random.uniform(10, 50),
                max_power_kw=random.uniform(2, 10),
                cycle_cost_dol_per_kwh=random.uniform(0.02, 0.08),
            )
        )
        all_agents.append(
            DeferrableLoadAgent(
                f"defer_{i}",
                power_kw=random.uniform(1, 3),
                duration_hours=2,
                preferred_start_hour=18,
                deferral_costs_dol={0: 0.1, 1: 0.5, 2: 1.5, 3: 5.0, 4: 15.0},
            )
        )  # 0hr deferral = run at preferred time
        all_agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=random.uniform(50, 150),
                duration_hours=4,
                preferred_window_hours=(0, 16),
                out_of_window_penalty_dol=random.uniform(75, 200),
            )
        )

    print("3. Collecting baselines and flexibility bids...")
    system_baseline_load = np.zeros(N_TIMESTEPS)
    all_flexibility_bids = []
    for agent in all_agents:
        system_baseline_load += agent.get_baseline_operation(sim_state)
        all_flexibility_bids.extend(agent.get_flexibility_bids(sim_state))
    print(f"Collected {len(all_flexibility_bids)} total bids.")

    print("4. Generating visualizations of available flexibility...")
    plot_flexibility_supply_curve(all_flexibility_bids)
    plot_cost_percentile_stack(all_flexibility_bids, N_TIMESTEPS)

    print("5. Running market clearing for peak shaving...")
    peak_threshold = np.percentile(system_baseline_load, 85)
    target_load = np.minimum(system_baseline_load, peak_threshold)
    final_achieved_load, dispatched_bids, total_cost = run_market_clearing(
        system_baseline_load, target_load, all_flexibility_bids
    )

    print("\n--- Simulation Results ---")
    print(
        f"System peak reduced from {np.max(system_baseline_load):.2f} kW to {np.max(final_achieved_load):.2f} kW."
    )
    print(
        f"Dispatched {len(dispatched_bids)} bids from {len(set(b['agent_id'] for b in dispatched_bids))} unique agents."
    )
    print(f"Total cost of flexibility procurement: ${total_cost:.2f}")

    print("6. Generating visualization of market dispatch results...")
    plot_market_dispatch_results(
        system_baseline_load, target_load, final_achieved_load, dispatched_bids
    )
    print("--- Simulation Complete ---")


if __name__ == "__main__":
    main()
