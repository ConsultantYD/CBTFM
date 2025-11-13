import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from faa_base import FAABase


class BuildingHVACAgent(FAABase):
    """
    REVISED HVAC Agent.
    - Bidding logic now calculates the true opportunity cost by comparing the
      total value of the entire baseline operational path vs. an alternative path.
    - This correctly prices in the long-term comfort consequences (thermal inertia)
      of a short-term flexibility action.
    """

    def __init__(
        self,
        agent_id: str,
        power_kw: float,
        r_val: float,
        c_val: float,
        comfort_cost_dol_per_deg_sq: float = 0.5,
        deadband_c: float = 0.5,
    ):
        super().__init__(agent_id, "HVAC")
        self.power_kw = power_kw
        self.r_val = r_val
        self.c_val = c_val
        self.comfort_cost = comfort_cost_dol_per_deg_sq
        self.deadband_c = deadband_c
        self.temp_c = 21.0

    def _get_path_value(
        self, power_schedule: np.ndarray, sim_state: Dict[str, Any]
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates the total value (negative cost) of a full operational path.
        Returns the total value and the resulting temperature trajectory.
        """
        prices = sim_state["price_dol_per_kwh"]
        setpoint = sim_state["hvac_setpoint_c"]

        temps = []
        current_temp = self.temp_c
        total_cost = 0.0

        for i, p in enumerate(power_schedule):
            # Simulate temperature evolution for this step
            delta_t_out = (sim_state["outdoor_temp_c"][i] - current_temp) / (
                self.r_val * self.c_val
            )
            delta_t_hvac = p / self.c_val
            current_temp += delta_t_out + delta_t_hvac
            temps.append(current_temp)

            # Calculate costs for this step
            discomfort_cost = (
                self.comfort_cost
                * max(0, abs(current_temp - setpoint) - self.deadband_c) ** 2
            )
            energy_cost = p * prices[i]
            total_cost += discomfort_cost + energy_cost

        return -total_cost, np.array(temps)

    # Convenience method for plotting temperature trajectories from a given schedule
    def _sim_temp(self, power_schedule: np.ndarray, sim_state: Dict[str, Any]) -> np.ndarray:
        _, temps = self._get_path_value(power_schedule, sim_state)
        return temps

    def get_baseline_operation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """Simple heuristic baseline: heat when temperature drops below the deadband."""
        outdoor = np.asarray(sim_state["outdoor_temp_c"], dtype=float)
        horizon = len(outdoor)
        baseline = np.zeros(horizon)
        temp = float(self.temp_c)
        setpoint = float(sim_state["hvac_setpoint_c"])

        is_heating = False
        for t in range(horizon):
            if is_heating:
                baseline[t] = self.power_kw
            elif temp < setpoint - self.deadband_c:
                baseline[t] = self.power_kw
                is_heating = True
            else:
                baseline[t] = 0.0

            delta_t_out = (outdoor[t] - temp) / (self.r_val * self.c_val)
            delta_t_hvac = baseline[t] / self.c_val
            temp += delta_t_out + delta_t_hvac

            if is_heating and temp >= setpoint:
                is_heating = False
        return baseline

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        """
        REVISED: Bids are priced as V(baseline) - V(alternative).
        This captures the full future impact of a deviation at a single hour.

        Premium options (applied only when cost > 0):
        - premium (fractional): offer_cost = cost * (1 + premium)
        - premium_abs (absolute $): offer_cost = cost + premium_abs
        Exactly one of these should be non-zero; if both > 0, a ValueError is raised.
        """
        baseline_power = self.get_baseline_operation(sim_state)
        baseline_value, _ = self._get_path_value(baseline_power, sim_state)
        bids = []
        horizon = len(baseline_power)

        for t in range(horizon):
            baseline_action = baseline_power[t]
            # Alternative action is to flip the state (ON->OFF or OFF->ON)
            alt_action = 0 if baseline_action > 0 else self.power_kw

            if abs(baseline_action - alt_action) < 1e-9:
                continue

            # Create the alternative path by changing the action at hour t
            alt_power = baseline_power.copy()
            alt_power[t] = alt_action

            # Calculate the total value of this new, full path
            alt_value, _ = self._get_path_value(alt_power, sim_state)

            # The cost is the loss in value from deviating from the baseline
            cost = baseline_value - alt_value
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

            bids.append(
                {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "cost": offer_cost,
                    "delta_p": alt_power - baseline_power,
                }
            )
        return bids


class EnergyStorageAgent(FAABase):
    """
    FAA for a BESS. Supports configurable charge/discharge power and duration.
    Generates per-hour bids by calculating the value of each primitive action
    (charge, idle, discharge) against a baseline that schedules charging on the
    lowest-price hours and discharging on the highest-price hours for
    `duration_hours` each.
    """

    def __init__(
        self,
        agent_id: str,
        capacity_kwh: float,
        max_power_kw: float,
        duration_hours: int = 1,
        efficiency: float = 0.9,
        cycle_cost_dol_per_kwh: float = 0.05,
    ):
        super().__init__(agent_id, "Storage")
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.duration_hours = int(duration_hours)
        self.efficiency = efficiency
        self.cycle_cost = cycle_cost_dol_per_kwh
        self.soc = 0.5

    def get_baseline_operation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        if prices.size == 0:
            return np.zeros(0)
        # Compute optimal arbitrage schedule with SOC constraints
        start_soc_kwh = float(self.soc) * float(self.capacity_kwh)
        value, p_schedule, _ = self._optimize_from(
            prices, start_t=0, start_soc_kwh=start_soc_kwh
        )
        return p_schedule

    def _get_action_values(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        # Deprecated: action values now computed via DP; keep signature for compatibility if referenced.
        zeros = np.zeros_like(prices, dtype=float)
        return {"charge": zeros, "idle": zeros, "discharge": zeros}

    def _optimize_from(
        self,
        prices: np.ndarray,
        start_t: int,
        start_soc_kwh: float,
        force_p: float = None,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Dynamic program to compute the optimal profit and schedule from start_t..end
        given an initial SOC (kWh). Optionally force the first action power (kW) at
        start_t to a specific value `force_p` (feasibility-checked w.r.t. SOC and limits).

        Returns (max_profit, power_profile, soc_profile) for the horizon slice.
        power_profile has length len(prices) - start_t. soc_profile has length len(prices) - start_t + 1.
        """
        prices = np.asarray(prices, dtype=float)
        n_total = len(prices)
        if start_t >= n_total:
            return 0.0, np.zeros(0), np.array([start_soc_kwh])

        P = float(self.max_power_kw)
        C = float(self.capacity_kwh)
        eff = float(self.efficiency) if self.efficiency > 0 else 1.0
        cc = float(self.cycle_cost)

        # Discretize SOC into levels
        num_levels = 101  # 1% resolution
        if C <= 0:
            # Degenerate storage: no capacity
            horizon = n_total - start_t
            return 0.0, np.zeros(horizon), np.array([0.0] * (horizon + 1))
        soc_grid = np.linspace(0.0, C, num_levels)
        step = soc_grid[1] - soc_grid[0]

        # Map starting SOC to nearest grid index
        s0_idx = int(np.clip(round(start_soc_kwh / step), 0, num_levels - 1))
        horizon = n_total - start_t

        # DP arrays
        V = np.full((horizon + 1, num_levels), -1e18, dtype=float)
        nxt = np.full((horizon, num_levels), -1, dtype=int)

        # Terminal value: zero for any SOC
        V[horizon, :] = 0.0

        # Iterate backwards
        for k in range(horizon - 1, -1, -1):
            t = start_t + k
            price = prices[t]
            for i, s in enumerate(soc_grid):
                # Compute reachable next SOC range given power limits and efficiency
                max_charge_ds = min(C - s, P * eff)  # Δs >= 0
                max_discharge_ds = min(s, P / eff)  # Δs >= 0 (we'll subtract)
                # Bounds in SOC index space
                j_min = int(np.ceil((s - max_discharge_ds) / step))
                j_max = int(np.floor((s + max_charge_ds) / step))
                j_min = max(j_min, 0)
                j_max = min(j_max, num_levels - 1)
                if j_min > j_max:
                    continue

                best_val = -1e18
                best_j = -1
                for j in range(j_min, j_max + 1):
                    s_next = soc_grid[j]
                    ds = s_next - s
                    if abs(ds) < 1e-12:
                        p = 0.0
                    elif ds > 0:
                        p = ds / eff  # Charge power (kW)
                        if p > P + 1e-9:
                            continue
                    else:
                        p = ds * eff  # Negative power (discharge)
                        if -p > P + 1e-9:
                            continue

                    # If we must force the first action, enforce at k == 0
                    if force_p is not None and k == 0:
                        # Enforce the first-step power with a small tolerance (kW)
                        if abs(p - force_p) > 1e-3:
                            continue

                    immediate = -price * p - cc * abs(p)
                    val = immediate + V[k + 1, j]
                    if val > best_val:
                        best_val = val
                        best_j = j

                V[k, i] = best_val
                nxt[k, i] = best_j

        # Recover schedule from s0_idx
        soc_profile = np.zeros(horizon + 1)
        power_profile = np.zeros(horizon)
        soc_profile[0] = soc_grid[s0_idx]
        cur_idx = s0_idx
        for k in range(horizon):
            j = int(nxt[k, cur_idx])
            if j < 0:
                # No feasible move; stay idle
                j = cur_idx
            s = soc_grid[cur_idx]
            s_next = soc_grid[j]
            ds = s_next - s
            if abs(ds) < 1e-12:
                p = 0.0
            elif ds > 0:
                p = ds / eff
            else:
                p = ds * eff
            power_profile[k] = p
            soc_profile[k + 1] = s_next
            cur_idx = j

        return float(V[0, s0_idx]), power_profile, soc_profile

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

        # Compute baseline optimal schedule under SOC constraints
        start_soc_kwh = float(self.soc) * float(self.capacity_kwh)
        base_value, base_p, base_soc = self._optimize_from(
            prices, start_t=0, start_soc_kwh=start_soc_kwh
        )

        bids: List[Dict] = []
        P = float(self.max_power_kw)
        C = float(self.capacity_kwh)
        eff = float(self.efficiency) if self.efficiency > 0 else 1.0

        # Helper to compute total profit for a given schedule
        def schedule_profit(p_schedule: np.ndarray) -> float:
            return float(
                np.sum(-prices * p_schedule - self.cycle_cost * np.abs(p_schedule))
            )

        # Walk through each hour and generate alternatives that are feasible given SOC
        cur_soc = start_soc_kwh
        for t in range(horizon):
            # SOC at start of hour t according to baseline
            cur_soc = base_soc[t]
            baseline_p_t = base_p[t]

            # Compute feasible alt powers among {-P, 0, +P}
            feasible_alts: List[float] = []
            # Charge +P feasibility: s_next = s + P*eff <= C
            if cur_soc + P * eff <= C + 1e-9:
                feasible_alts.append(+P)
            # Idle always feasible
            feasible_alts.append(0.0)
            # Discharge -P feasibility: s_next = s - P/eff >= 0
            if cur_soc - P / eff >= -1e-9:
                feasible_alts.append(-P)

            # Generate bids for alternatives different from baseline action
            for alt_p in feasible_alts:
                if abs(alt_p - baseline_p_t) < 1e-9:
                    continue

                # Optimize remainder from t+1 given the SOC after applying alt_p at t
                if alt_p >= 0:
                    next_soc = min(C, cur_soc + alt_p * eff)
                else:
                    next_soc = max(0.0, cur_soc + alt_p / eff)

                # Immediate profit effect at time t
                immediate = -prices[t] * alt_p - self.cycle_cost * abs(alt_p)

                rem_value, rem_p, rem_soc = self._optimize_from(
                    prices, start_t=t + 1, start_soc_kwh=next_soc
                )

                # New schedule assembled: prefix baseline up to t-1, forced alt at t, optimized suffix
                new_p = np.copy(base_p)
                new_p[t] = alt_p
                if t + 1 < horizon:
                    new_p[t + 1 :] = rem_p

                new_profit = (
                    immediate
                    + rem_value
                    + float(
                        np.sum(
                            -prices[:t] * new_p[:t]
                            - self.cycle_cost * np.abs(new_p[:t])
                        )
                    )
                )

                cost = base_value - new_profit
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
                delta_p = new_p - base_p

                # Only include bids that actually change something and have a finite cost
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


class DeferrableLoadAgent(FAABase):
    """
    REVISED Deferrable Load Agent.
    - Bidding logic now finds the single best (minimum cost) schedule first.
    - All other possible schedules are then priced as bids, with the cost being
      the true opportunity cost: (Cost_of_Alternative - Cost_of_Best).
    """

    def __init__(
        self,
        agent_id: str,
        power_kw: float,
        duration_hours: int,
        preferred_start_hour: int,
        deferral_costs_dol: Dict[int, float],
        deadline_hour: Optional[int] = None,
    ):
        super().__init__(agent_id, "Deferrable")
        self.power = power_kw
        self.duration = int(duration_hours)
        self.pref_start = int(preferred_start_hour)
        # Allow None for backward compatibility; resolve at runtime using price horizon
        self.deadline = None if deadline_hour is None else int(deadline_hour)
        self.deferral_costs = deferral_costs_dol

    def _calc_run_cost(self, start_time: int, sim_state: Dict[str, Any]) -> float:
        """Simple cost function: energy cost + inconvenience cost."""
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        end_time = start_time + self.duration
        effective_deadline = (
            self.deadline if self.deadline is not None else len(prices)
        )

        if start_time < 0 or end_time > len(prices) or end_time > effective_deadline:
            return float("inf")

        energy_cost = float(np.sum(prices[start_time:end_time]) * self.power)
        offset = abs(int(start_time - self.pref_start))
        inconvenience_cost = float(self.deferral_costs.get(offset, float("inf")))
        return energy_cost + inconvenience_cost

    def get_baseline_operation(self, sim_state: Dict[str, Any]) -> np.ndarray:
        """The baseline is the single best schedule with the minimum total cost."""
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        effective_deadline = (
            self.deadline if self.deadline is not None else horizon
        )

        possible_times = range(0, max(0, effective_deadline - self.duration + 1))
        costs = {t: self._calc_run_cost(t, sim_state) for t in possible_times}

        if not costs or min(costs.values()) == float("inf"):
            return np.zeros(horizon)

        best_start_time = min(costs, key=costs.get)
        baseline = np.zeros(horizon)
        baseline[best_start_time : best_start_time + self.duration] = self.power
        return baseline

    def get_flexibility_bids(
        self,
        sim_state: Dict[str, Any],
        premium: float = 0.0,
        premium_abs: float = 0.0,
    ) -> List[Dict]:
        """
        REVISED: Bids are priced as Cost(alternative) - Cost(best).
        This is the true opportunity cost of choosing a suboptimal schedule.

        Premium options (applied only when cost > 0):
        - premium (fractional): offer_cost = cost * (1 + premium)
        - premium_abs (absolute $): offer_cost = cost + premium_abs
        Exactly one of these should be non-zero; if both > 0, a ValueError is raised.
        """
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        horizon = len(prices)
        effective_deadline = (
            self.deadline if self.deadline is not None else horizon
        )

        possible_times = range(0, max(0, effective_deadline - self.duration + 1))
        all_costs = {t: self._calc_run_cost(t, sim_state) for t in possible_times}

        valid_costs = {t: c for t, c in all_costs.items() if np.isfinite(c)}
        if not valid_costs:
            return []

        best_start_time = min(valid_costs, key=valid_costs.get)
        min_total_cost = valid_costs[best_start_time]

        baseline_power = np.zeros(horizon)
        baseline_power[best_start_time : best_start_time + self.duration] = self.power

        bids: List[Dict] = []
        for t, alt_cost in valid_costs.items():
            if t == best_start_time:
                continue

            opportunity_cost = alt_cost - min_total_cost
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


class CILoadShiftAgent(DeferrableLoadAgent):
    """C&I load-shift agent with backward-compatible constructor.

    Supports two calling styles:
    1) Dictionary deferral costs (newer Deferrable semantics):
       CILoadShiftAgent(id, power_kw, duration_hours, preferred_start_hour, deferral_costs_dol, deadline_hour=None)

    2) Preferred window + outside-window penalty (older examples 2/3):
       CILoadShiftAgent(id, power_kw, duration_hours, preferred_window_hours=(lo, hi), out_of_window_penalty_dol=pen)
    """

    def __init__(
        self,
        agent_id: str,
        power_kw: float,
        duration_hours: int,
        preferred_start_hour: Optional[int] = None,
        deferral_costs_dol: Optional[Dict[int, float]] = None,
        deadline_hour: Optional[int] = None,
        *,
        preferred_window_hours: Optional[Tuple[int, int]] = None,
        out_of_window_penalty_dol: Optional[float] = None,
    ):
        self._ci_window: Optional[Tuple[int, int]] = None
        self._ci_penalty: float = 0.0

        if preferred_window_hours is not None:
            # Older calling style: window + penalty. Defer to parent with empty deferral costs
            self._ci_window = (int(preferred_window_hours[0]), int(preferred_window_hours[1]))
            self._ci_penalty = float(out_of_window_penalty_dol or 0.0)
            if preferred_start_hour is None:
                # Use the start of the window as a harmless placeholder
                preferred_start_hour = self._ci_window[0]
            super().__init__(
                agent_id=agent_id,
                power_kw=power_kw,
                duration_hours=duration_hours,
                preferred_start_hour=preferred_start_hour,
                deferral_costs_dol={},
                deadline_hour=deadline_hour,
            )
        else:
            # Newer calling style: explicit deferral costs dictionary
            super().__init__(
                agent_id=agent_id,
                power_kw=power_kw,
                duration_hours=duration_hours,
                preferred_start_hour=int(preferred_start_hour or 0),
                deferral_costs_dol=deferral_costs_dol or {},
                deadline_hour=deadline_hour,
            )
        self.agent_type = "C&I"

    def _calc_run_cost(self, start_time: int, sim_state: Dict[str, Any]) -> float:
        if self._ci_window is None:
            # Fall back to DeferrableLoadAgent behavior
            return super()._calc_run_cost(start_time, sim_state)

        # Window-based penalty semantics used by older examples
        prices = np.asarray(sim_state["price_dol_per_kwh"], dtype=float)
        end_time = start_time + self.duration
        effective_deadline = (
            self.deadline if self.deadline is not None else len(prices)
        )
        if start_time < 0 or end_time > len(prices) or end_time > effective_deadline:
            return float("inf")

        energy_cost = float(np.sum(prices[start_time:end_time]) * self.power)
        lo, hi = self._ci_window
        latest_valid_start = hi - self.duration + 1
        in_window = (start_time >= lo) and (start_time <= latest_valid_start)
        penalty = 0.0 if in_window else self._ci_penalty
        return energy_cost + penalty
