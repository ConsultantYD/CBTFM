import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from faa_base import FAABase
from faa_agents import (
    EnergyStorageAgent,
    BuildingHVACAgent,
    CILoadShiftAgent,
)


class EnergyStorageAgentHeuristic(EnergyStorageAgent):
    """
    Heuristic version of the BESS FAA.

    - Keeps the same baseline computation (via DP) for a good reference path.
    - Speeds up bid pricing by using a one-step lookahead approximation:
      Q(action at t) = immediate_profit(t) + V_hat(next_state at t+1).
    - V_hat is a fast linear heuristic that uses precomputed forward-looking
      price summaries shared across agents.
    """

    def __init__(
        self,
        agent_id: str,
        capacity_kwh: float,
        max_power_kw: float,
        duration_hours: int = 1,
        efficiency: float = 0.9,
        cycle_cost_dol_per_kwh: float = 0.05,
        lookups: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            capacity_kwh=capacity_kwh,
            max_power_kw=max_power_kw,
            duration_hours=duration_hours,
            efficiency=efficiency,
            cycle_cost_dol_per_kwh=cycle_cost_dol_per_kwh,
        )
        self._lookups = lookups

    @staticmethod
    def precompute_lookups(prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Precompute forward-looking price summaries (shared across agents):
        - future_max_price[t] = max(prices[t:])
        - future_min_price[t] = min(prices[t:])

        Returned arrays have length == len(prices). These are intended to be
        used with indices t+1 where applicable; callers should handle the
        boundary (t == horizon-1) by treating the future value as zero.
        """
        p = np.asarray(prices, dtype=float)
        n = len(p)
        if n == 0:
            return {"future_max_price": np.array([]), "future_min_price": np.array([])}

        future_max = np.empty(n, dtype=float)
        future_min = np.empty(n, dtype=float)
        running_max = -np.inf
        running_min = np.inf
        for k in range(n - 1, -1, -1):
            running_max = max(running_max, p[k])
            running_min = min(running_min, p[k])
            future_max[k] = running_max
            future_min[k] = running_min
        return {"future_max_price": future_max, "future_min_price": future_min}

    def _get_future_value(self, soc_kwh: float, t: int, prices: np.ndarray) -> float:
        """
        Heuristic value of being at SOC (kWh) at time index t.
        Uses shared forward-looking max/min prices as a proxy for arbitrage margin.

        V_hat(s, t) = alpha(t) * s, where
          alpha(t) ≈ eff * max_future_price[t] - (1/eff) * min_future_price[t]
                     - cycle_cost * (eff + 1/eff)

        For boundary t >= horizon, returns 0.
        """
        p = np.asarray(prices, dtype=float)
        n = len(p)
        if t >= n:
            return 0.0

        if self._lookups is None:
            self._lookups = self.precompute_lookups(p)

        # For one-step lookahead at time t+1, callers will pass t+1. Still guard bounds.
        idx = int(t)
        if idx >= n:
            return 0.0

        eff = float(self.efficiency) if self.efficiency > 0 else 1.0
        fmax = float(self._lookups["future_max_price"][idx])
        fmin = float(self._lookups["future_min_price"][idx])
        margin = eff * fmax - (1.0 / eff) * fmin - self.cycle_cost * (eff + 1.0 / eff)
        alpha = max(margin, 0.0)
        s = float(np.clip(soc_kwh, 0.0, float(self.capacity_kwh)))
        return alpha * s

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        if horizon == 0:
            return []

        # Ensure lookups are available (shared across agents ideally)
        if self._lookups is None:
            self._lookups = self.precompute_lookups(prices)

        # Baseline via DP for a strong reference schedule
        start_soc_kwh = float(self.soc) * float(self.capacity_kwh)
        base_value, base_p, base_soc = self._optimize_from(
            prices, start_t=0, start_soc_kwh=start_soc_kwh
        )

        bids: List[Dict] = []
        P = float(self.max_power_kw)
        C = float(self.capacity_kwh)
        eff = float(self.efficiency) if self.efficiency > 0 else 1.0

        for t in range(horizon):
            s = float(base_soc[t])
            p_base = float(base_p[t])

            # Feasible alternative actions among {-P, 0, +P}
            feasible_alts: List[float] = []
            if s + P * eff <= C + 1e-9:
                feasible_alts.append(+P)
            feasible_alts.append(0.0)
            if s - P / eff >= -1e-9:
                feasible_alts.append(-P)

            # Baseline next-state and immediate term
            if p_base >= 0:
                s_next_base = min(C, s + p_base * eff)
            else:
                s_next_base = max(0.0, s + p_base / eff)
            immediate_base = -prices[t] * p_base - self.cycle_cost * abs(p_base)
            q_base = immediate_base + self._get_future_value(s_next_base, t + 1, prices)

            for p_alt in feasible_alts:
                if abs(p_alt - p_base) < 1e-9:
                    continue

                if p_alt >= 0:
                    s_next_alt = min(C, s + p_alt * eff)
                else:
                    s_next_alt = max(0.0, s + p_alt / eff)
                immediate_alt = -prices[t] * p_alt - self.cycle_cost * abs(p_alt)
                q_alt = immediate_alt + self._get_future_value(s_next_alt, t + 1, prices)

                # Heuristic opportunity cost
                cost = q_base - q_alt
                offer_cost = float(cost)
                # Clamp negative costs to zero so bids are not discarded upstream
                if offer_cost < 0:
                    offer_cost = 0.0
                if offer_cost > 0:
                    if premium > 0.0 and premium_abs > 0.0:
                        raise ValueError(
                            "Provide only one of premium (fractional) or premium_abs (absolute)."
                        )
                    if premium_abs > 0.0:
                        offer_cost = offer_cost + float(premium_abs)
                    elif premium > 0.0:
                        offer_cost = offer_cost * (1.0 + float(premium))

                # Only a one-hour deviation in delta_p for the heuristic
                delta_p = np.zeros(horizon)
                delta_p[t] = p_alt - p_base

                if np.any(np.abs(delta_p) > 1e-9) and np.isfinite(cost):
                    bids.append(
                        {
                            "agent_id": self.agent_id,
                            "agent_type": self.agent_type,
                            "cost": offer_cost,
                            "delta_p": delta_p,
                        }
                    )

        return bids


class BuildingHVACAgentHeuristic(BuildingHVACAgent):
    """
    Heuristic HVAC FAA using one-step lookahead.

    - Avoids re-simulating the entire thermal path per bid.
    - For each hour, compares Q = -immediate_cost + V_hat(next_temp, t+1)
      between the baseline action and its single-step alternative (flip ON/OFF).
    """

    def _get_future_value(self, temp_c: float, t: int, sim_state: Dict[str, Any]) -> float:
        """
        Simple heuristic future value at time index t for a given indoor temp.
        Penalizes deviation from setpoint using the same quadratic comfort cost.
        Returns a negative penalty to be added to immediate value (i.e., value = -penalty).
        """
        # Only use setpoint and deadband; keep it lightweight
        setpoint = float(sim_state["hvac_setpoint_c"])
        discomfort = max(0.0, abs(float(temp_c) - setpoint) - float(self.deadband_c))
        penalty = float(self.comfort_cost) * (discomfort ** 2)
        # At or beyond horizon end, treat as no future penalty
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        if t >= len(prices):
            return 0.0
        return -penalty

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        outdoor = np.asarray(sim_state["outdoor_temp_c"], dtype=float)
        setpoint = float(sim_state["hvac_setpoint_c"])
        horizon = len(prices)
        if horizon == 0:
            return []

        baseline = self.get_baseline_operation(sim_state)
        bids: List[Dict] = []

        temp = float(self.temp_c)
        for t in range(horizon):
            p_base = float(baseline[t])
            p_alt = 0.0 if p_base > 1e-9 else float(self.power_kw)
            if abs(p_base - p_alt) < 1e-9:
                # No alternative if already the same
                # Advance physical state under baseline and continue
                delta_t_out = (outdoor[t] - temp) / (self.r_val * self.c_val)
                delta_t_hvac = p_base / self.c_val
                temp = temp + delta_t_out + delta_t_hvac
                continue

            # One-step state updates for baseline and alternative
            delta_t_out = (outdoor[t] - temp) / (self.r_val * self.c_val)
            next_temp_base = temp + delta_t_out + p_base / self.c_val
            next_temp_alt = temp + delta_t_out + p_alt / self.c_val

            # Immediate costs (energy + discomfort at next_temp)
            discomfort_base = max(0.0, abs(next_temp_base - setpoint) - self.deadband_c)
            discomfort_alt = max(0.0, abs(next_temp_alt - setpoint) - self.deadband_c)
            immediate_cost_base = prices[t] * p_base + self.comfort_cost * (discomfort_base ** 2)
            immediate_cost_alt = prices[t] * p_alt + self.comfort_cost * (discomfort_alt ** 2)

            # Heuristic future value term uses next state's temp at t+1
            q_base = -immediate_cost_base + self._get_future_value(next_temp_base, t + 1, sim_state)
            q_alt = -immediate_cost_alt + self._get_future_value(next_temp_alt, t + 1, sim_state)

            cost = q_base - q_alt
            offer_cost = float(cost)
            # Clamp negative costs to zero so bids are not discarded upstream
            if offer_cost < 0:
                offer_cost = 0.0
            if offer_cost > 0:
                if premium > 0.0 and premium_abs > 0.0:
                    raise ValueError(
                        "Provide only one of premium (fractional) or premium_abs (absolute)."
                    )
                if premium_abs > 0.0:
                    offer_cost = offer_cost + float(premium_abs)
                elif premium > 0.0:
                    offer_cost = offer_cost * (1.0 + float(premium))

            delta_p = np.zeros(horizon)
            delta_p[t] = p_alt - p_base
            if np.any(np.abs(delta_p) > 1e-9) and np.isfinite(cost):
                bids.append(
                    {
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type,
                        "cost": offer_cost,
                        "delta_p": delta_p,
                    }
                )

            # Advance temp along the baseline path for the next hour
            temp = next_temp_base

        return bids


# Top-level helper; must be picklable for multiprocessing on some platforms
def _calc_ci_run_cost_worker(
    start_time: int,
    prices: np.ndarray,
    power_kw: float,
    duration_hours: int,
    effective_deadline: int,
    mode: str,
    preferred_start_hour: int,
    deferral_costs_dol: Dict[int, float],
    window_lo: int,
    window_hi: int,
    out_of_window_penalty_dol: float,
):
    # Validate feasibility
    end_time = start_time + duration_hours
    if (
        start_time < 0
        or end_time > len(prices)
        or end_time > effective_deadline
    ):
        return (start_time, float("inf"))

    energy_cost = float(np.sum(prices[start_time:end_time]) * float(power_kw))

    if mode == "window":
        latest_valid_start = int(window_hi) - int(duration_hours) + 1
        in_window = (start_time >= int(window_lo)) and (start_time <= latest_valid_start)
        penalty = 0.0 if in_window else float(out_of_window_penalty_dol)
        return (start_time, energy_cost + penalty)

    # mode == "deferral"
    offset = abs(int(start_time) - int(preferred_start_hour))
    inconvenience_cost = float(deferral_costs_dol.get(offset, float("inf")))
    return (start_time, energy_cost + inconvenience_cost)


class CILoadShiftAgentHeuristic(CILoadShiftAgent):
    """
    Heuristic C&I agent. Preserves the correct opportunity-cost logic, but
    parallelizes the initial cost computation across all feasible start times
    using multiprocessing.Pool.starmap.
    """

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        import multiprocessing as mp
        import os

        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        effective_deadline = self.deadline if self.deadline is not None else horizon

        possible_times = list(range(0, max(0, effective_deadline - self.duration + 1)))
        if not possible_times:
            return []

        # Determine calculation mode and parameters for workers
        if getattr(self, "_ci_window", None) is not None:
            mode = "window"
            window_lo, window_hi = self._ci_window
            pref_start = int(getattr(self, "pref_start", 0))
            def_costs = {}
            out_pen = float(getattr(self, "_ci_penalty", 0.0))
        else:
            mode = "deferral"
            window_lo, window_hi = 0, 0
            pref_start = int(self.pref_start)
            def_costs = dict(self.deferral_costs)
            out_pen = 0.0

        # Serial fallback for very small search spaces where Pool overhead dominates
        if len(possible_times) <= 8:
            all_cost_pairs = [
                _calc_ci_run_cost_worker(
                    t,
                    prices,
                    float(self.power),
                    int(self.duration),
                    int(effective_deadline),
                    mode,
                    int(pref_start),
                    def_costs,
                    int(window_lo),
                    int(window_hi),
                    float(out_pen),
                )
                for t in possible_times
            ]
        else:
            # Parallel path
            # Cap workers to avoid oversubscription in many-agent settings
            procs = min(len(possible_times), max(1, (os.cpu_count() or 2) - 1))
            with mp.Pool(processes=procs) as pool:
                all_cost_pairs = pool.starmap(
                    _calc_ci_run_cost_worker,
                    [
                        (
                            t,
                            prices,
                            float(self.power),
                            int(self.duration),
                            int(effective_deadline),
                            mode,
                            int(pref_start),
                            def_costs,
                            int(window_lo),
                            int(window_hi),
                            float(out_pen),
                        )
                        for t in possible_times
                    ],
                )

        all_costs = {t: c for (t, c) in all_cost_pairs}

        valid_costs = {t: c for t, c in all_costs.items() if np.isfinite(c)}
        if not valid_costs:
            return []

        best_start_time = min(valid_costs, key=valid_costs.get)
        min_total_cost = float(valid_costs[best_start_time])

        baseline_power = np.zeros(horizon)
        baseline_power[best_start_time : best_start_time + self.duration] = self.power

        bids: List[Dict] = []
        for t, alt_cost in valid_costs.items():
            if t == best_start_time:
                continue

            opportunity_cost = float(alt_cost) - float(min_total_cost)
            offer_cost = float(opportunity_cost)
            if offer_cost > 0:
                if premium > 0.0 and premium_abs > 0.0:
                    raise ValueError(
                        "Provide only one of premium (fractional) or premium_abs (absolute)."
                    )
                if premium_abs > 0.0:
                    offer_cost = offer_cost + float(premium_abs)
                elif premium > 0.0:
                    offer_cost = offer_cost * (1.0 + float(premium))

            alt_power = np.zeros(horizon)
            alt_power[t : t + self.duration] = self.power

            bids.append(
                {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "cost": offer_cost,
                    "delta_p": alt_power - baseline_power,
                }
            )

        return bids


class EnergyStorageAgentHeuristicFast(EnergyStorageAgentHeuristic):
    """
    Even faster heuristic variant for BESS:
    - Replaces DP baseline with a greedy threshold schedule (O(T)).
    - Keeps one-step lookahead opportunity-cost pricing with V_hat.
    - Shares the same precomputed price lookups.
    """

    def _greedy_baseline(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices = np.asarray(prices, dtype=float)
        T = len(prices)
        P = float(self.max_power_kw)
        C = float(self.capacity_kwh)
        eff = float(self.efficiency) if self.efficiency > 0 else 1.0

        p_sched = np.zeros(T, dtype=float)
        soc = np.zeros(T + 1, dtype=float)
        soc[0] = float(self.soc) * C

        for t in range(T):
            s = soc[t]
            # Horizon-aware dual thresholds using forward prices
            pf = prices[t:]
            if pf.size > 0:
                q_low = float(np.quantile(pf, 0.25))
                q_high = float(np.quantile(pf, 0.80))
                margin = float(self.cycle_cost) * (eff + 1.0 / eff)
                if q_high - q_low < margin:
                    mid = 0.5 * (q_high + q_low)
                    q_low = mid - 0.5 * margin
                    q_high = mid + 0.5 * margin
            else:
                q_low = q_high = float(prices[t])

            # Try to charge on low prices if room remains; discharge on high prices
            if prices[t] <= q_low + 1e-12 and s + P * eff <= C + 1e-9:
                p = P
            elif prices[t] >= q_high - 1e-12 and s - P / eff >= -1e-9:
                p = -P
            else:
                p = 0.0
            # Enforce SOC bounds exactly
            if p >= 0:
                p = min(p, (C - s) / max(eff, 1e-9))
                s_next = s + p * eff
            else:
                p = -min(-p, s * max(eff, 1e-9))
                s_next = s + p / eff
            p_sched[t] = p
            soc[t + 1] = float(np.clip(s_next, 0.0, C))

        return p_sched, soc

    def get_baseline_operation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        p_sched, _ = self._greedy_baseline(prices)
        return p_sched

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        if horizon == 0:
            return []

        if self._lookups is None:
            self._lookups = self.precompute_lookups(prices)

        base_p, base_soc = self._greedy_baseline(prices)
        P = float(self.max_power_kw)
        C = float(self.capacity_kwh)
        eff = float(self.efficiency) if self.efficiency > 0 else 1.0

        bids: List[Dict] = []
        for t in range(horizon):
            s = float(base_soc[t])
            p_base = float(base_p[t])

            feasible_alts: List[float] = []
            if s + P * eff <= C + 1e-9:
                feasible_alts.append(+P)
            feasible_alts.append(0.0)
            if s - P / eff >= -1e-9:
                feasible_alts.append(-P)

            # Baseline Q
            if p_base >= 0:
                s_next_base = min(C, s + p_base * eff)
            else:
                s_next_base = max(0.0, s + p_base / eff)
            immediate_base = -prices[t] * p_base - self.cycle_cost * abs(p_base)
            q_base = immediate_base + self._get_future_value(s_next_base, t + 1, prices)

            for p_alt in feasible_alts:
                if abs(p_alt - p_base) < 1e-9:
                    continue

                if p_alt >= 0:
                    s_next_alt = min(C, s + p_alt * eff)
                else:
                    s_next_alt = max(0.0, s + p_alt / eff)
                immediate_alt = -prices[t] * p_alt - self.cycle_cost * abs(p_alt)
                q_alt = immediate_alt + self._get_future_value(s_next_alt, t + 1, prices)

                cost = q_base - q_alt
                offer_cost = float(cost)
                if offer_cost > 0:
                    if premium > 0.0 and premium_abs > 0.0:
                        raise ValueError(
                            "Provide only one of premium (fractional) or premium_abs (absolute)."
                        )
                    if premium_abs > 0.0:
                        offer_cost = offer_cost + float(premium_abs)
                    elif premium > 0.0:
                        offer_cost = offer_cost * (1.0 + float(premium))

                delta_p = np.zeros(horizon)
                delta_p[t] = p_alt - p_base
                if np.any(np.abs(delta_p) > 1e-9) and np.isfinite(cost):
                    bids.append(
                        {
                            "agent_id": self.agent_id,
                            "agent_type": self.agent_type,
                            "cost": offer_cost,
                            "delta_p": delta_p,
                        }
                    )

        return bids


class CILoadShiftAgentHeuristicFast(CILoadShiftAgent):
    """
    Faster C&I heuristic using vectorized convolution for energy cost instead of
    multiprocessing. Preserves opportunity cost logic.
    """

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        effective_deadline = self.deadline if self.deadline is not None else horizon
        d = int(self.duration)
        if horizon == 0 or d <= 0 or d > horizon:
            return []

        # Valid start indices under length and deadline
        max_start = max(0, min(horizon - d, effective_deadline - d))
        if max_start < 0:
            return []
        starts = np.arange(0, max_start + 1, dtype=int)

        # Vectorized energy costs via convolution
        window = np.ones(d, dtype=float)
        energy_segment_sums = np.convolve(prices, window, mode="valid")[: max_start + 1]
        energy_costs = energy_segment_sums * float(self.power)

        if getattr(self, "_ci_window", None) is not None:
            lo, hi = self._ci_window
            latest_valid_start = hi - d + 1
            in_window = (starts >= int(lo)) & (starts <= int(latest_valid_start))
            penalties = np.where(in_window, 0.0, float(self._ci_penalty))
        else:
            pref = int(self.pref_start)
            # Map offsets to penalties (fills missing with inf)
            def_pen = np.full_like(starts, np.inf, dtype=float)
            for off, val in self.deferral_costs.items():
                idx = np.where(np.abs(starts - pref) == int(off))[0]
                if idx.size > 0:
                    def_pen[idx] = float(val)
            penalties = def_pen

        total_costs = energy_costs + penalties
        valid_mask = np.isfinite(total_costs)
        if not np.any(valid_mask):
            return []
        starts = starts[valid_mask]
        total_costs = total_costs[valid_mask]

        # Pick best start and produce opportunity-cost bids
        best_idx = int(np.argmin(total_costs))
        best_start = int(starts[best_idx])
        min_total_cost = float(total_costs[best_idx])

        baseline = np.zeros(horizon)
        baseline[best_start : best_start + d] = float(self.power)

        bids: List[Dict] = []
        for t, alt_cost in zip(starts, total_costs):
            t = int(t)
            if t == best_start:
                continue
            opportunity_cost = float(alt_cost) - float(min_total_cost)
            offer_cost = float(opportunity_cost)
            if offer_cost > 0:
                if premium > 0.0 and premium_abs > 0.0:
                    raise ValueError(
                        "Provide only one of premium (fractional) or premium_abs (absolute)."
                    )
                if premium_abs > 0.0:
                    offer_cost = offer_cost + float(premium_abs)
                elif premium > 0.0:
                    offer_cost = offer_cost * (1.0 + float(premium))

            alt = np.zeros(horizon)
            alt[t : t + d] = float(self.power)
            bids.append(
                {
                    "agent_id": self.agent_id,
                    "agent_type": "C&I",
                    "cost": offer_cost,
                    "delta_p": alt - baseline,
                }
            )

        return bids
