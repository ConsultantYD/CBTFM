import numpy as np
import pandas as pd
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faa_agents import (
    BuildingHVACAgent,
    EnergyStorageAgent,
    CILoadShiftAgent,
)
from visualizations import plot_market_dispatch_results
from market_operator import MarketOperator


def main():
    """
    This example demonstrates the full end-to-end market simulation:
    1. A system baseline is established from the fleet of agents.
    2. A system-level objective (peak shaving) is defined.
    3. The market clearing algorithm is run to dispatch the most cost-effective
       agents to meet that objective.
    4. The final results are analyzed and visualized.
    """
    print("--- Example 3: Full Market Clearing Simulation ---")
    N_TIMESTEPS = 24
    N_AGENTS_PER_TYPE = 50

    # Bid premium controls (min event profitability premium by asset type)
    PREMIUM_STORAGE = 0.10
    PREMIUM_HVAC = 0.10
    PREMIUM_CI = 0.10

    # 1. Setup Environment and Agents (same as Example 2)
    print("1. Creating simulation environment and instantiating agent fleet...")
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, N_TIMESTEPS)))
    prices[17:20] *= 1.5
    sim_state = {
        "outdoor_temp_c": 10
        + 10 * np.sin(np.linspace(0, 2 * np.pi, N_TIMESTEPS) - np.pi / 1.5),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }

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
        # C&I with randomized duration and preferred window to avoid stacking at t=0
        ci_duration = int(random.randint(2, 6))
        lo = int(random.randint(0, max(0, N_TIMESTEPS - ci_duration)))
        hi_min = lo + ci_duration - 1
        hi_max = N_TIMESTEPS - 1
        hi = int(random.randint(hi_min, min(hi_min + 8, hi_max)))
        all_agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=random.uniform(50, 150),
                duration_hours=ci_duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=random.uniform(75, 200),
            )
        )

    # 2. Calculate System Baseline and Collect All Bids
    print("2. Calculating system baseline and collecting all bids...")
    system_baseline_load = np.zeros(N_TIMESTEPS)
    all_flexibility_bids = []
    for agent in all_agents:
        system_baseline_load += agent.get_baseline_operation(sim_state)
        type_to_premium = {
            "Storage": PREMIUM_STORAGE,
            "HVAC": PREMIUM_HVAC,
            "C&I": PREMIUM_CI,
        }
        prem = type_to_premium.get(getattr(agent, "agent_type", ""), 0.0)
        all_flexibility_bids.extend(agent.get_flexibility_bids(sim_state, premium=prem))

    print(f"System Baseline Peak Load: {np.max(system_baseline_load):.2f} kW")
    print(f"Collected {len(all_flexibility_bids)} total bids from the fleet.")

    # 3. Define the System Objective: Peak Shaving
    print("\n3. Defining system objective: Shave the peak load...")
    peak_threshold = np.percentile(system_baseline_load, 85)
    target_load = np.minimum(system_baseline_load, peak_threshold)
    print(f"Peak shaving target set at {peak_threshold:.2f} kW.")

    # 4. Run the Market Clearing Algorithm
    print("4. Running the CBTFM clearing algorithm to dispatch bids...")
    mo = MarketOperator(peak_guard_factor=1.05)
    final_achieved_load, dispatched_bids, total_cost = mo.clear_market(
        all_bids=all_flexibility_bids,
        system_baseline_load=system_baseline_load,
        target_load_shape=target_load,
    )

    # 5. Analyze and Print the Results
    print("\n--- Step 5: Market Simulation Results ---")
    print(f"Final Achieved Peak Load: {np.max(final_achieved_load):.2f} kW")
    peak_reduction = np.max(system_baseline_load) - np.max(final_achieved_load)
    print(f"Peak Reduction Achieved: {peak_reduction:.2f} kW")
    print(f"Total Cost of Flexibility Procurement: ${total_cost:.2f}")

    dispatched_agents = set(b["agent_id"] for b in dispatched_bids)
    print(
        f"Dispatched {len(dispatched_bids)} bids from {len(dispatched_agents)} unique agents."
    )

    print("\n--- Dispatched Agent Breakdown ---")
    df_dispatch = pd.DataFrame(dispatched_bids)
    if not df_dispatch.empty:
        # Calculate the net energy change (kWh) for each dispatched bid
        df_dispatch["net_kwh_change"] = df_dispatch["delta_p"].apply(np.sum)
        # Display counts and average cost per agent type
        type_summary = (
            df_dispatch.groupby("agent_type")
            .agg(
                num_dispatched=("agent_id", "nunique"),
                avg_cost=("cost", "mean"),
                total_kwh_shifted=("net_kwh_change", lambda x: np.sum(np.abs(x))),
            )
            .round(2)
        )
        print(type_summary)

    # 6. Visualize the Final Dispatch
    print("\n6. Generating final visualization of the market outcome...")
    plot_market_dispatch_results(
        system_baseline_load, target_load, final_achieved_load, dispatched_bids
    )
    # Bonus: Same bids, different target shapes
    print("\n--- Bonus: Same bids, different target shapes ---")
    alt_targets = {}
    # Cap at the 75th percentile of baseline
    cap75 = np.minimum(system_baseline_load, np.percentile(system_baseline_load, 75))
    alt_targets["75th percentile cap"] = cap75
    # Flat 15% reduction on hours above median
    med = np.median(system_baseline_load)
    flat15 = system_baseline_load.copy()
    mask = flat15 > med
    flat15[mask] = flat15[mask] * 0.85
    alt_targets["Flat -15% above median"] = flat15

    for name, tgt in alt_targets.items():
        alt_achieved, alt_dispatched, alt_cost = mo.clear_market(
            all_bids=all_flexibility_bids,
            system_baseline_load=system_baseline_load,
            target_load_shape=tgt,
        )
        print(f"\nTarget: {name}")
        print(
            f"Peak: baseline {np.max(system_baseline_load):.2f} → target {np.max(tgt):.2f} → achieved {np.max(alt_achieved):.2f} kW"
        )
        print(f"Dispatched {len(alt_dispatched)} bids | Total cost ${alt_cost:.2f}")

    print("--- Simulation Complete ---")


if __name__ == "__main__":
    main()
