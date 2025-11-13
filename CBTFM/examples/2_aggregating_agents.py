import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faa_agents import (
    BuildingHVACAgent,
    EnergyStorageAgent,
    CILoadShiftAgent,
)
from visualizations import plot_flexibility_supply_curve, plot_cost_percentile_stack

# ================================
# Controls — tweak scenario quickly
# ================================
N_TIMESTEPS = 24
N_AGENTS_PER_TYPE = 50
STACK_WINDOW_START = 18
STACK_WINDOW_END = 21

# Bid premium (min profitability margin). Applied only to positive-cost bids.
PREMIUM_STORAGE = 0.10
PREMIUM_HVAC = 0.10
PREMIUM_DEFERRABLE = 0.10
PREMIUM_CI = 0.10

# Agent parameter ranges
HVAC_POWER_RANGE = (5.0, 15.0)
HVAC_COMFORT_COST_RANGE = (0.4, 1.0)

STOR_CAP_RANGE = (10.0, 50.0)
STOR_PWR_RANGE = (2.0, 10.0)
STOR_CYCLE_COST_RANGE = (0.02, 0.08)

CI_POWER_RANGE = (1.0, 5.0)
# Randomize C&I job duration and eligible window per agent to avoid stacking
CI_DURATION_RANGE = (2, 6)  # hours
CI_PENALTY_RANGE = (75.0, 200.0)


def main():
    """
    This example shows how the bids from a diverse fleet of agents are
    aggregated to create a system-level view of the available flexibility.
    """
    print("--- Example 2: Fleet Aggregation and Visualization ---")

    # 1. Create a realistic simulation environment
    prices = 0.10 + 0.15 * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, N_TIMESTEPS)))
    prices[17:20] *= 1.5
    sim_state = {
        "outdoor_temp_c": 10
        + 10 * np.sin(np.linspace(0, 2 * np.pi, N_TIMESTEPS) - np.pi / 1.5),
        "hvac_setpoint_c": 21.0,
        "price_dol_per_kwh": prices,
    }

    # 2. Instantiate a fleet of heterogeneous agents
    print(
        f"Instantiating {N_AGENTS_PER_TYPE * 3} total agents with varied parameters..."
    )
    all_agents = []
    for i in range(N_AGENTS_PER_TYPE):
        all_agents.append(
            BuildingHVACAgent(
                f"hvac_{i}",
                power_kw=random.uniform(*HVAC_POWER_RANGE),
                r_val=2,
                c_val=10,
                comfort_cost_dol_per_deg_sq=random.uniform(*HVAC_COMFORT_COST_RANGE),
            )
        )
        all_agents.append(
            EnergyStorageAgent(
                f"bess_{i}",
                capacity_kwh=random.uniform(*STOR_CAP_RANGE),
                max_power_kw=random.uniform(*STOR_PWR_RANGE),
                cycle_cost_dol_per_kwh=random.uniform(*STOR_CYCLE_COST_RANGE),
            )
        )
        # C&I: randomize job duration and preferred start window per agent
        ci_duration = int(random.randint(*CI_DURATION_RANGE))
        # choose a random window [lo, hi] that can accommodate the duration
        lo = int(random.randint(0, max(0, N_TIMESTEPS - ci_duration)))
        # ensure hi >= lo + duration - 1
        hi_min = lo + ci_duration - 1
        hi_max = N_TIMESTEPS - 1
        # add extra slack up to 8 hours beyond duration when possible
        hi = int(random.randint(hi_min, min(hi_min + 8, hi_max)))
        all_agents.append(
            CILoadShiftAgent(
                f"ci_{i}",
                power_kw=random.uniform(*CI_POWER_RANGE),
                duration_hours=ci_duration,
                preferred_window_hours=(lo, hi),
                out_of_window_penalty_dol=random.uniform(*CI_PENALTY_RANGE),
            )
        )

    # 3. Collect all bids from all agents into a single list
    print("Collecting all flexibility bids from the entire fleet...")
    all_flexibility_bids = []
    type_premium = {
        "Storage": PREMIUM_STORAGE,
        "HVAC": PREMIUM_HVAC,
        "Deferrable": PREMIUM_DEFERRABLE,
        "C&I": PREMIUM_CI,
    }
    for agent in all_agents:
        prem = type_premium.get(getattr(agent, "agent_type", ""), 0.0)
        all_flexibility_bids.extend(agent.get_flexibility_bids(sim_state, premium=prem))
    print(f"Collected a total of {len(all_flexibility_bids)} bids.")
    print("This list represents the total potential flexibility of the system.")

    # 4. Generate visualizations from the aggregated data
    print("\n--- Step 4: Visualizing the Aggregated Flexibility Resource ---")
    print("The following plots will be saved as PDF files in 'results/'.")

    # Visualization 1: The Flexibility Supply Curve
    print("\nGenerating Supply Curve...")
    print(
        "This plot shows 'How much total load reduction can we buy at a certain price?'"
    )
    print(
        "It aggregates all bids across all times to show the pure cost-quantity trade-off."
    )
    plot_flexibility_supply_curve(
        all_flexibility_bids,
        window_start=STACK_WINDOW_START,
        window_end=STACK_WINDOW_END,
    )

    # Visualization 2: The Cost Percentile Stack
    print("\nGenerating Cost Percentile Stack...")
    print(
        "This plot shows 'WHEN is flexibility available and what is its quality (cost)?'"
    )
    print(
        "It shows the total available load reduction at each hour, color-coded by cost."
    )
    plot_cost_percentile_stack(
        all_flexibility_bids,
        N_TIMESTEPS,
        num_percentile_steps=5,
        window_start=STACK_WINDOW_START,
        window_end=STACK_WINDOW_END,
    )


if __name__ == "__main__":
    main()
