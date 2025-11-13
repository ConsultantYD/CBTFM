# examples/example_1_single_agent_bidding.py
import numpy as np
import pandas as pd
from typing import Optional
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faa_agents import (
    EnergyStorageAgent,
    BuildingHVACAgent,
    CILoadShiftAgent,
)

# This example requires its own visualization functions, which we've moved to visualizations.py
# Make sure your main visualizations.py is in the parent directory
from visualizations import (
    plot_storage_bid_story,
    plot_generic_bid_story,
    plot_hvac_temp_bid_story,
    plot_baseline_with_bid_cloud,
)

# ================================
# Flexibility event controls (hrs)
# Adjust these to test different event windows
# ================================
STORAGE_EVENT_START = 12
STORAGE_EVENT_LEN = 2
CI_EVENT_START = 14
CI_EVENT_LEN = 4
HVAC_EVENT_START = 18
HVAC_EVENT_LEN = 2

# ================================
# Minimum profitability premium controls
# - For storage in this example, we use an absolute premium ($): offer_cost = cost + PREMIUM_STORAGE_ABS
# - For other assets, we keep fractional premium: offer_cost = cost * (1 + premium)
# ================================
PREMIUM_STORAGE_ABS = 1.00  # absolute $ adder applied to positive-cost bids
PREMIUM_CI = 0.0  # % of cost (cost * (1 + PREMIUM_CI))
PREMIUM_HVAC = 0.0  # % of cost (cost * (1 + PREMIUM_HVAC))

# ================================
# Tunable agent parameters (playground)
# Update these to quickly try different asset behaviors
# ================================
# Storage
STORAGE_CAP_KWH = 40.0
STORAGE_MAX_POWER_KW = 10.0
STORAGE_DURATION_H = 4
STORAGE_EFFICIENCY = 1.0
STORAGE_CYCLE_COST = 0.05
STORAGE_INIT_SOC = 0.50  # fraction of capacity

# Industrial C&I process
CI_POWER_KW = 80.0
CI_DURATION_H = 4
CI_PREF_START_HOUR = 12
CI_DEFERRAL_COSTS = {0: 0.0, 1: 80.0, 2: 180.0, 3: 320.0}

# HVAC
HVAC_POWER_KW = 8.0
HVAC_R_VAL = 2.0
HVAC_C_VAL = 10.0
HVAC_COMFORT_COST = 0.7
HVAC_DEADBAND_C = 1.0


def pick_min_cost_reduction_bid(bids, baseline_power_kw):
    """
    Select the minimum-cost bid that yields a reduction in grid consumption
    for non-storage assets. Grid convention: negative = load (withdrawal).
    Returns None if no such bid exists.
    """
    if not bids:
        return None
    base_p = np.asarray(baseline_power_kw, dtype=float)
    grid_base = -np.abs(base_p)
    best = None
    best_cost = float("inf")
    for b in bids:
        delta = np.asarray(b["delta_p"], dtype=float)
        new_p = base_p + delta
        grid_new = -np.abs(new_p)
        delta_grid = grid_new - grid_base
        # Enforce earliest change hour >= 2
        reduction_mask = delta_grid > 1e-9
        if not np.any(reduction_mask):
            continue
        # Require at least one reduction hour at t >= 2
        valid_after2 = np.any(reduction_mask & (np.arange(len(reduction_mask)) >= 2))
        if not valid_after2:
            continue
        if b["cost"] < best_cost:
            best = b
            best_cost = b["cost"]
    return best


def pick_min_cost_reduction_bid_storage(bids, baseline_power_kw):
    """
    Select the minimum-cost bid for storage that reduces grid consumption
    (grid moves toward zero, i.e., Δgrid > 0 at any hour).
    Grid convention: negative = load. Returns None if no such bid exists.
    """
    if not bids:
        return None
    base_p = np.asarray(baseline_power_kw, dtype=float)
    grid_base = -base_p  # storage: grid = -power
    best = None
    best_cost = float("inf")
    for b in bids:
        delta = np.asarray(b["delta_p"], dtype=float)
        new_p = base_p + delta
        grid_new = -new_p
        delta_grid = grid_new - grid_base
        # Enforce earliest change hour >= 2
        reduction_mask = delta_grid > 1e-9
        if not np.any(reduction_mask):
            continue
        # Require at least one reduction hour at t >= 2
        valid_after2 = np.any(reduction_mask & (np.arange(len(reduction_mask)) >= 2))
        if not valid_after2:
            continue
        if b["cost"] < best_cost:
            best = b
            best_cost = b["cost"]
    return best


def pick_best_reduction_bid(bids, baseline_power_kw, prices):
    """
    Choose the reduction bid that maximizes price-weighted reduction benefit
    (ignores bid cost), requiring at least one reduction hour at t ≥ 2.
    - Reduction benefit = sum over hours of price[$/kWh] * Δgrid[kW] where Δgrid > 0.
    - Grid convention for non-storage: grid = -abs(power).
    Returns None if no reduction at t≥2 exists.
    """
    if not bids:
        return None
    base_p = np.asarray(baseline_power_kw, dtype=float)
    prices = np.asarray(prices, dtype=float)
    grid_base = -np.abs(base_p)
    best = None
    best_score = -1e18
    best_benefit = 0.0
    for b in bids:
        delta = np.asarray(b["delta_p"], dtype=float)
        new_p = base_p + delta
        grid_new = -np.abs(new_p)
        delta_grid = grid_new - grid_base
        mask_idx = np.arange(len(delta_grid)) >= 2
        reduction = np.where((delta_grid > 1e-9) & mask_idx, delta_grid, 0.0)
        if np.all(reduction == 0.0):
            continue
        benefit = float(np.sum(prices * reduction))
        score = benefit  # ignore bid cost
        # Prefer higher benefit; tie-break on lower cost for determinism
        if (score > best_score) or (
            abs(score - best_score) < 1e-9 and (benefit > best_benefit)
        ):
            best = b
            best_score = score
            best_benefit = benefit
    return best


def _event_mask(
    n: int, start: int = None, length: int = None, *, min_hour: int = 2
) -> np.ndarray:
    idx = np.arange(n)
    mask = idx >= min_hour
    if start is not None and length is not None and length > 0:
        in_win = (idx >= start) & (idx < start + length)
        mask = mask & in_win
    return mask


def pick_best_reduction_bid_load(
    bids,
    baseline_power_kw,
    prices,
    *,
    event_start: Optional[int] = None,
    event_len: Optional[int] = None,
    min_hour: int = 2,
):
    """
    Select the bid for a pure load that maximizes price-weighted reduction
    benefit within the specified event window (if provided), minus bid cost.
    Grid convention for loads: grid = -abs(power).
    """
    if not bids:
        return None
    base_p = np.asarray(baseline_power_kw, dtype=float)
    prices = np.asarray(prices, dtype=float)
    grid_base = -np.abs(base_p)
    mask = _event_mask(len(base_p), event_start, event_len, min_hour=min_hour)

    best = None
    best_score = -1e18
    best_benefit = 0.0
    for b in bids:
        delta = np.asarray(b["delta_p"], dtype=float)
        new_p = base_p + delta
        grid_new = -np.abs(new_p)
        delta_grid = grid_new - grid_base
        reduction = np.where((delta_grid > 1e-9) & mask, delta_grid, 0.0)
        if np.all(reduction == 0.0):
            continue
        benefit = float(np.sum(prices * reduction))
        score = benefit  # ignore bid cost
        if (score > best_score) or (
            abs(score - best_score) < 1e-9 and benefit > best_benefit
        ):
            best = b
            best_score = score
            best_benefit = benefit
    return best


def pick_best_reduction_bid_storage(
    bids,
    baseline_power_kw,
    prices,
    *,
    event_start: Optional[int] = None,
    event_len: Optional[int] = None,
    min_hour: int = 2,
):
    """
    Select the storage bid that maximizes price-weighted reduction within
    the event window (if provided), minus bid cost. Storage grid = -power.
    """
    if not bids:
        return None
    base_p = np.asarray(baseline_power_kw, dtype=float)
    prices = np.asarray(prices, dtype=float)
    grid_base = -base_p
    mask = _event_mask(len(base_p), event_start, event_len, min_hour=min_hour)

    best = None
    best_score = -1e18
    best_benefit = 0.0
    for b in bids:
        delta = np.asarray(b["delta_p"], dtype=float)
        new_p = base_p + delta
        grid_new = -new_p
        delta_grid = grid_new - grid_base
        reduction = np.where((delta_grid > 1e-9) & mask, delta_grid, 0.0)
        if np.all(reduction == 0.0):
            continue
        benefit = float(np.sum(prices * reduction))
        score = benefit  # ignore bid cost
        if (score > best_score) or (
            abs(score - best_score) < 1e-9 and benefit > best_benefit
        ):
            best = b
            best_score = score
            best_benefit = benefit
    return best


def explain_storage_bids(hour, bids, baseline_power_kw, prices):
    """Helper to explain storage bids in asset convention."""
    bids_for_hour = [b for b in bids if abs(b["delta_p"][hour]) > 1e-9]
    print(f"\n--- Analysis for Hour {hour} (Price: ${prices[hour]:.2f}/kWh) ---")
    print(
        f"Agent's Baseline Action: {baseline_power_kw[hour]:.2f} kW (+charge, -discharge)"
    )

    if not bids_for_hour:
        print("No flexibility bids generated; agent is following optimal schedule.")
        return

    for bid in bids_for_hour:
        delta_kw = bid["delta_p"][hour]
        final_power = baseline_power_kw[hour] + delta_kw
        action_desc = (
            "Charge" if final_power > 0 else "Discharge" if final_power < 0 else "Idle"
        )

        print(f"\n  Offer to change state to: '{action_desc}'")
        print(f"    - Deviation from Baseline: {delta_kw:.2f} kW")
        print(f"    - Cost of this deviation: ${bid['cost']:.2f}")
        print(
            "    - Explanation: The cost is the total profit lost compared to the perfect baseline schedule."
        )


def run_storage_example():
    """Deep dive into the EnergyStorageAgent's bidding logic."""
    print("--- Example 1a: Energy Storage Agent Deep Dive ---")
    prices = np.array(
        [
            0.10,
            0.08,
            0.05,
            0.06,
            0.12,
            0.18,
            0.25,
            0.22,
            0.15,
            0.11,
            0.09,
            0.10,
            0.13,
            0.17,
            0.24,
            0.30,
            0.35,
            0.40,
            0.32,
            0.28,
            0.22,
            0.18,
            0.15,
            0.12,
        ]
    )
    sim_state = {"price_dol_per_kwh": prices}

    agent = EnergyStorageAgent(
        agent_id="bess_001",
        capacity_kwh=STORAGE_CAP_KWH,
        max_power_kw=STORAGE_MAX_POWER_KW,
        duration_hours=STORAGE_DURATION_H,
        efficiency=STORAGE_EFFICIENCY,
        cycle_cost_dol_per_kwh=STORAGE_CYCLE_COST,
    )
    # Set initial SOC (fraction) for the storage agent
    agent.soc = STORAGE_INIT_SOC
    print(
        f"Created Agent: {agent.agent_id} (Capacity: {agent.capacity_kwh} kWh, Power: {agent.max_power_kw} kW)"
    )

    baseline_power = agent.get_baseline_operation(sim_state)

    print("\n--- Step 1: Agent's Optimal Baseline Operation (Asset Convention) ---")
    charge_hours = np.where(baseline_power > 0)[0]
    discharge_hours = np.where(baseline_power < 0)[0]
    print(
        f"Optimal Charge Hours: {charge_hours.tolist()} @ Prices: {[f'${p:.2f}' for p in prices[charge_hours]]}"
    )
    print(
        f"Optimal Discharge Hours: {discharge_hours.tolist()} @ Prices: {[f'${p:.2f}' for p in prices[discharge_hours]]}"
    )

    flexibility_bids = agent.get_flexibility_bids(
        sim_state,
        premium_abs=PREMIUM_STORAGE_ABS,
    )
    # Plot baseline and all alternative profiles implied by bids
    # plot_baseline_with_bid_cloud(
    #     baseline_power_kw=baseline_power,
    #     bids=flexibility_bids,
    #     agent_id=agent.agent_id,
    #     tag=f"bid_cloud_{agent.agent_id}",
    #     y_title="Power (kW; +charge, -discharge)",
    # )
    print(f"\n--- Step 2: Generating All {len(flexibility_bids)} Flexibility Bids ---")

    explain_storage_bids(
        hour=8, bids=flexibility_bids, baseline_power_kw=baseline_power, prices=prices
    )
    explain_storage_bids(
        hour=charge_hours[0],
        bids=flexibility_bids,
        baseline_power_kw=baseline_power,
        prices=prices,
    )
    explain_storage_bids(
        hour=discharge_hours[0],
        bids=flexibility_bids,
        baseline_power_kw=baseline_power,
        prices=prices,
    )

    if flexibility_bids:
        best_bid = pick_best_reduction_bid_storage(
            flexibility_bids,
            baseline_power,
            prices,
            event_start=STORAGE_EVENT_START,
            event_len=STORAGE_EVENT_LEN,
            min_hour=2,
        )
        if best_bid is not None:
            print(
                f"\n--- Step 3: Visualizing max-benefit (ignoring cost) load-reduction bid (Bid cost=${best_bid['cost']:.2f}) ---"
            )
            plot_storage_bid_story(
                prices=prices,
                baseline_power_kw=baseline_power,
                bid_delta_p=best_bid["delta_p"],
                capacity_kwh=agent.capacity_kwh,
                efficiency=agent.efficiency,
                init_soc_frac=agent.soc,
                cycle_cost_dol_per_kwh=agent.cycle_cost,
                agent_id=agent.agent_id,
                tag=f"mincost_reduction_{agent.agent_id}",
                bid_cost=float(best_bid["cost"]),
                event_len_hours=STORAGE_EVENT_LEN,
                event_start_hour=STORAGE_EVENT_START,
            )
        else:
            print("No storage bids reduce grid consumption; skipping storage plot.")


def run_ci_example():
    """Industrial C&I load-shift bid story (dictionary displacement costs)."""
    print("\n\n--- Example 1b: Industrial C&I Load Shift ---")
    prices = np.array(
        [
            0.10,
            0.08,
            0.05,
            0.06,
            0.12,
            0.18,
            0.25,
            0.22,
            0.15,
            0.11,
            0.09,
            0.10,
            0.13,
            0.17,
            0.24,
            0.30,
            0.35,
            0.40,
            0.32,
            0.28,
            0.22,
            0.18,
            0.15,
            0.12,
        ]
    )

    # C&I Load Shift Example — dictionary-based displacement costs by |offset|
    ci = CILoadShiftAgent(
        agent_id="ci_demo",
        power_kw=CI_POWER_KW,
        duration_hours=CI_DURATION_H,
        preferred_start_hour=CI_PREF_START_HOUR,
        deferral_costs_dol=CI_DEFERRAL_COSTS,
    )
    ci_base = ci.get_baseline_operation({"price_dol_per_kwh": prices})
    ci_bids = ci.get_flexibility_bids({"price_dol_per_kwh": prices}, premium=PREMIUM_CI)
    # Plot baseline and all alternative profiles implied by bids
    # plot_baseline_with_bid_cloud(
    #     baseline_power_kw=ci_base,
    #     bids=ci_bids,
    #     agent_id=ci.agent_id,
    #     tag=f"bid_cloud_{ci.agent_id}",
    #     y_title="Power (kW; positive = load)",
    # )
    if ci_bids:
        best_bid = pick_best_reduction_bid_load(
            ci_bids,
            ci_base,
            prices,
            event_start=CI_EVENT_START,
            event_len=CI_EVENT_LEN,
            min_hour=2,
        )
        if best_bid is not None:
            print(
                f"\n--- Visualizing max-benefit (ignoring cost) load-reduction bid for C&I Load (Bid cost=${best_bid['cost']:.2f}) ---"
            )
            plot_generic_bid_story(
                prices=prices,
                baseline_power_kw=ci_base,
                bid_delta_p=best_bid["delta_p"],
                agent_id=ci.agent_id,
                tag=f"mincost_reduction_{ci.agent_id}",
                bid_cost=float(best_bid["cost"]),
                event_len_hours=CI_EVENT_LEN,
                event_start_hour=CI_EVENT_START,
            )
        else:
            print("No C&I bids reduce grid consumption at t>=2; skipping plot.")


def run_hvac_example():
    """HVAC bid story (2h event, reduction-focused)."""
    print("\n\n--- Example 1c: HVAC Flexibility Bid Story ---")
    prices = np.array(
        [
            0.10,
            0.08,
            0.05,
            0.06,
            0.12,
            0.18,
            0.25,
            0.22,
            0.15,
            0.11,
            0.09,
            0.10,
            0.13,
            0.17,
            0.24,
            0.30,
            0.35,
            0.40,
            0.32,
            0.28,
            0.22,
            0.18,
            0.15,
            0.12,
        ]
    )
    hvac = BuildingHVACAgent(
        agent_id="hvac_demo",
        power_kw=HVAC_POWER_KW,
        r_val=HVAC_R_VAL,
        c_val=HVAC_C_VAL,
        comfort_cost_dol_per_deg_sq=HVAC_COMFORT_COST,
        deadband_c=HVAC_DEADBAND_C,
    )
    hvac_sim = {
        "price_dol_per_kwh": prices,
        "outdoor_temp_c": 10
        + 10 * np.sin(np.linspace(0, 2 * np.pi, len(prices)) - np.pi / 1.5),
        "hvac_setpoint_c": 21.0,
    }
    hvac_base = hvac.get_baseline_operation(hvac_sim)
    hvac_bids_all = hvac.get_flexibility_bids(hvac_sim, premium=PREMIUM_HVAC)
    # Plot baseline and all alternative profiles implied by bids
    # plot_baseline_with_bid_cloud(
    #     baseline_power_kw=hvac_base,
    #     bids=hvac_bids_all,
    #     agent_id=hvac.agent_id,
    #     tag=f"bid_cloud_{hvac.agent_id}",
    #     y_title="Power (kW; positive = heating)",
    # )

    # Try with configured event window first
    best_bid = pick_best_reduction_bid_load(
        hvac_bids_all,
        hvac_base,
        prices,
        event_start=HVAC_EVENT_START,
        event_len=HVAC_EVENT_LEN,
        min_hour=2,
    )

    chosen_event_start = HVAC_EVENT_START
    chosen_event_len = HVAC_EVENT_LEN

    # Fallback: auto-detect earliest heating hour >= 2 to ensure a demonstrative plot
    if best_bid is None:
        heating_hours = np.where((hvac_base > 1e-9) & (np.arange(len(prices)) >= 2))[0]
        if heating_hours.size > 0:
            chosen_event_start = int(heating_hours[0])
            chosen_event_len = HVAC_EVENT_LEN
            best_bid = pick_best_reduction_bid_load(
                hvac_bids_all,
                hvac_base,
                prices,
                event_start=chosen_event_start,
                event_len=chosen_event_len,
                min_hour=2,
            )
        else:
            # Last resort: allow any hour (still t>=2) to find a load-reduction demonstration
            best_bid = pick_best_reduction_bid_load(
                hvac_bids_all, hvac_base, prices, min_hour=2
            )
            chosen_event_start = None

    if best_bid is not None:
        print(f"Creating HVAC bid story (load reduction, cost=${best_bid['cost']:.2f})")
        hvac_new_p = hvac_base + best_bid["delta_p"]
        # Use start-of-hour temperatures to match storage semantics (impact shows next hour)
        base_end_t = hvac._sim_temp(hvac_base, hvac_sim)
        new_end_t = hvac._sim_temp(hvac_new_p, hvac_sim)
        init_t = float(hvac.temp_c)
        base_t = np.concatenate(([init_t], base_end_t[:-1]))
        new_t = np.concatenate(([init_t], new_end_t[:-1]))
        setpoint_series = np.full_like(prices, hvac_sim["hvac_setpoint_c"], dtype=float)
        plot_hvac_temp_bid_story(
            prices=prices,
            baseline_power_kw=hvac_base,
            bid_delta_p=best_bid["delta_p"],
            baseline_temp_c=base_t,
            with_bid_temp_c=new_t,
            setpoint_c=setpoint_series,
            agent_id=hvac.agent_id,
            tag=f"mincost_reduction_{hvac.agent_id}",
            bid_cost=float(best_bid["cost"]),
            event_len_hours=HVAC_EVENT_LEN,
            event_start_hour=chosen_event_start,
        )
    else:
        print(
            "No HVAC bids reduce grid consumption at t>=2 under any window; skipping plot."
        )


if __name__ == "__main__":
    run_storage_example()
    run_ci_example()
    run_hvac_example()
