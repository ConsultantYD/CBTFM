"""
Market Operator for the Cost-Based Transactive Flexibility Market (CBTFM).

This module defines the MarketOperator class, which encapsulates the market
clearing logic previously implemented as a standalone function
(`run_market_clearing`). The Market Operator (MO) receives flexibility bids
from agents and determines a dispatch that best meets a requested target load
shape (e.g., peak shaving) when applied to the system baseline load.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np


class MarketOperator:
    """
    Market Operator for CBTFM clearing.

    The MarketOperator performs a greedy, single-shot clearing that respects a
    "one bid per agent" constraint. At each iteration it selects the available
    bid that provides the largest reduction in squared error relative to a
    target load shape, per unit cost, while avoiding creation of materially
    worse new peaks.

    Parameters
    ----------
    peak_guard_factor : float, optional
        Guard factor to prevent creating new peaks while applying a bid.
        A candidate bid is skipped if the proposed maximum load exceeds
        `peak_guard_factor * current_max_load`. Default is 1.05 (i.e., no more
        than +5% over the current achieved peak).
    """

    def __init__(self, *, peak_guard_factor: float = 1.05) -> None:
        self.peak_guard_factor = float(peak_guard_factor)

    def clear_market(
        self,
        all_bids: List[Dict],
        system_baseline_load: np.ndarray,
        target_load_shape: np.ndarray,
        *,
        focus_windows: Optional[List[Tuple[int, int]]] = None,
        max_bids_per_agent: int = 1,
        max_bids_per_agent_by_type: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Clear the CBTFM by dispatching a subset of bids to best match the
        target load shape when applied to the baseline load.

        The algorithm is greedy and enforces that at most one bid is dispatched
        per agent by default. You can relax this with `max_bids_per_agent`, or
        provide per-type limits via `max_bids_per_agent_by_type`.
        Bids are chosen by the ratio of (clamped) cost to squared-error
        reduction, subject to a guard
        that prevents creating new load peaks above a configurable threshold.

        Parameters
        ----------
        all_bids : List[Dict]
            The list of flexibility bids. Each bid is a dictionary expected to
            contain keys: "agent_id", "agent_type" (optional for reporting),
            "delta_p" (np.ndarray-like power delta), and "cost" (float).
        system_baseline_load : np.ndarray
            Baseline system load profile (kW) to which bid deltas will be
            applied.
        target_load_shape : np.ndarray
            Target load profile (kW) representing the desired shape the Market
            Operator tries to match (e.g., peak shaving cap).

        Optional Parameters
        -------------------
        focus_windows : Optional[List[Tuple[int, int]]]
            A list of inclusive (start, end) index pairs specifying the hours
            over which the clearing objective is evaluated. When provided, the
            squared-error objective and the peak guard are restricted to these
            hours; hours outside the union of windows are ignored (the MO does
            not attempt to shape them). If None, the objective uses the full
            horizon.

        Returns
        -------
        final_achieved_load : np.ndarray
            The achieved system load after applying the dispatched bids.
        dispatched_bids : List[Dict]
            The subset of bids selected by the clearing algorithm.
        total_cost : float
            The total cost of the dispatched bids.
        """
        achieved_load = np.asarray(system_baseline_load, dtype=float).copy()
        target_load = np.asarray(target_load_shape, dtype=float)

        # Build objective mask from focus windows (inclusive indices)
        if focus_windows:
            mask = np.zeros_like(achieved_load, dtype=bool)
            n = achieved_load.size
            for (s, e) in focus_windows:
                if n == 0:
                    continue
                lo = max(0, int(s))
                hi = min(n - 1, int(e))
                if hi < lo:
                    lo, hi = hi, lo
                mask[lo : hi + 1] = True
        else:
            mask = np.ones_like(achieved_load, dtype=bool)

        # Index bids by agent; include all bids but pre-filter to reduce unrelated offers
        available_bids: Dict[str, List[Dict]] = {}
        # Determine objective direction within the focus window for coarse filtering
        down_request = bool(np.all(target_load[mask] <= achieved_load[mask])) if np.any(mask) else False
        up_request = bool(np.all(target_load[mask] >= achieved_load[mask])) if np.any(mask) else False

        tiny = 1e-9
        for bid in all_bids:
            agent_id = bid.get("agent_id")
            if agent_id is None:
                continue
            delta_p = np.asarray(bid.get("delta_p", []), dtype=float)
            if delta_p.shape != achieved_load.shape:
                continue
            # Filter: ignore bids with zero impact in the focus window
            impact = np.sum(np.abs(delta_p[mask])) if np.any(mask) else np.sum(np.abs(delta_p))
            if impact <= tiny:
                continue
            # Directional filter: if all requested changes are down, require any negative in mask; vice versa for up
            if down_request and not np.any(delta_p[mask] < -tiny):
                continue
            if up_request and not np.any(delta_p[mask] > tiny):
                continue
            available_bids.setdefault(agent_id, []).append(bid)

        dispatched_bids: List[Dict] = []
        total_cost: float = 0.0
        dispatched_count: Dict[str, int] = {}

        # Greedy selection
        while True:
            best_bid = None
            best_score = float("inf")
            best_new_load = None

            current_error = float(np.sum(((achieved_load - target_load)[mask]) ** 2))
            current_peak = float(np.max(achieved_load[mask])) if np.any(mask) else 0.0
            peak_guard = self.peak_guard_factor * current_peak

            for agent_id, bids in available_bids.items():
                for bid in bids:
                    delta_p = np.asarray(bid.get("delta_p", []), dtype=float)
                    if delta_p.shape != achieved_load.shape:
                        continue
                    proposed_load = achieved_load + delta_p

                    # Peak guard: skip if this bid creates a materially higher peak
                    if np.any(mask) and float(np.max(proposed_load[mask])) > peak_guard:
                        continue

                    new_error = float(
                        np.sum(((proposed_load - target_load)[mask]) ** 2)
                    )
                    error_reduction = current_error - new_error
                    if error_reduction <= 1e-6:
                        continue

                    bid_cost = float(bid.get("cost", 0.0))
                    eff_cost = max(0.0, bid_cost)
                    score = eff_cost / error_reduction if error_reduction > 0 else float("inf")

                    if score < best_score:
                        best_score = score
                        best_bid = bid
                        best_new_load = proposed_load

            if best_bid is None:
                break

            # Commit the best bid and update remaining bids for that agent
            achieved_load = best_new_load  # type: ignore[assignment]
            # Clamp negative costs to zero for settlement accounting
            total_cost += max(0.0, float(best_bid.get("cost", 0.0)))
            dispatched_bids.append(best_bid)
            agent_id = best_bid.get("agent_id")
            agent_type = best_bid.get("agent_type")
            dispatched_count[agent_id] = dispatched_count.get(agent_id, 0) + 1
            # Determine cap for this agent (type-specific cap overrides global)
            cap = int(max_bids_per_agent)
            if max_bids_per_agent_by_type is not None and agent_type in max_bids_per_agent_by_type:
                cap = int(max_bids_per_agent_by_type[agent_type])
            # Remove only the selected bid for this agent
            remaining = [b for b in available_bids.get(agent_id, []) if b is not best_bid]
            if dispatched_count[agent_id] >= cap or not remaining:
                if agent_id in available_bids:
                    del available_bids[agent_id]
            else:
                available_bids[agent_id] = remaining

        return achieved_load, dispatched_bids, total_cost
