# visualizations.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional


def plot_storage_bid_story(
    prices,
    baseline_power_kw,
    bid_delta_p,
    capacity_kwh,
    efficiency,
    init_soc_frac,
    cycle_cost_dol_per_kwh,
    agent_id,
    tag,
    bid_cost,
    event_len_hours,
    event_start_hour: Optional[int] = None,
):
    """
    Creates a detailed plot for a single Energy Storage bid, showing power, SOC, and profit.
    """
    base_p = np.asarray(baseline_power_kw, dtype=float)
    delta = np.asarray(bid_delta_p, dtype=float)
    bid_p = base_p + delta

    # Calculate SOC profiles
    init_soc = init_soc_frac * capacity_kwh
    base_soc = np.zeros(len(base_p) + 1)
    bid_soc = np.zeros(len(bid_p) + 1)
    base_soc[0] = init_soc
    bid_soc[0] = init_soc
    for t in range(len(base_p)):
        base_soc[t + 1] = np.clip(
            base_soc[t]
            + (base_p[t] * efficiency if base_p[t] > 0 else base_p[t] / efficiency),
            0,
            capacity_kwh,
        )
        bid_soc[t + 1] = np.clip(
            bid_soc[t]
            + (bid_p[t] * efficiency if bid_p[t] > 0 else bid_p[t] / efficiency),
            0,
            capacity_kwh,
        )

    # Calculate profit profiles
    base_hp = -prices * base_p - cycle_cost_dol_per_kwh * np.abs(base_p)
    bid_hp = -prices * bid_p - cycle_cost_dol_per_kwh * np.abs(bid_p)
    diff_hp = bid_hp - base_hp
    base_profit = float(np.sum(base_hp))
    bid_profit = float(np.sum(bid_hp))

    # 4-row layout: Price, Grid Power, SOC, Profit (bottom row uses twin axes)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.18, 0.32, 0.26, 0.24],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )

    x = list(range(len(prices)))
    # Row 1: Price
    fig.add_trace(
        go.Scatter(x=x, y=prices, name="Price", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )

    # Row 2: Grid Power (grid = -power; negative = load)
    grid_base = -base_p
    grid_bid = -bid_p
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_base,
            name="Baseline",
            legendgroup="baseline",
            showlegend=True,
            line=dict(color="black", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_bid,
            name="Flex Dispatched Power",
            line=dict(color="#2ca02c", width=3),
        ),
        row=2,
        col=1,
    )
    # Highlight configured event window if provided; else auto-detect reduction
    n = len(prices)
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        s = max(0, int(event_start_hour))
        e = min(n - 1, s + int(event_len_hours) - 1)
        x0 = s - 0.5
        x1 = e + 0.5
        for rr in (1, 2, 3, 4):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(255,165,0,0.12)",
                line_width=0,
                row=rr,
                col=1,
            )
    else:
        delta_grid = grid_bid - grid_base
        reduction_mask = delta_grid > 1e-9
        mask_idx = np.arange(len(delta_grid))
        reduction_mask = reduction_mask & (mask_idx >= 2)
        if np.any(reduction_mask):
            idx = np.where(reduction_mask)[0]
            starts = [idx[0]]
            ends = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    ends.append(idx[i - 1])
                    starts.append(idx[i])
            ends.append(idx[-1])
            for s, e in zip(starts, ends):
                x0 = s - 0.5
                x1 = e + 0.5
                for rr in (1, 2, 3, 4):
                    fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor="rgba(255,165,0,0.12)",
                        line_width=0,
                        row=rr,
                        col=1,
                    )
    # Row 3: SOC (start-of-hour state; impact of control appears next hour)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=base_soc[:-1] / capacity_kwh * 100,
            name="Baseline",
            legendgroup="baseline",
            showlegend=False,
            line=dict(color="black", dash="dash"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bid_soc[:-1] / capacity_kwh * 100,
            name="Flex Dispatched SOC",
            line=dict(color="#9467bd", width=3),
        ),
        row=3,
        col=1,
    )
    # Determine where to allocate the dispatch payment:
    # Prefer the first hour AFTER the configured event window; otherwise fallback to
    # first reduction hour, else first hour with any change.
    delta = bid_p - base_p
    delta_grid = (-bid_p) - (-base_p)
    red_mask = delta_grid > 1e-9
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        pay_t = min(len(prices) - 1, int(event_start_hour) + int(event_len_hours))
    else:
        pay_t = (
            int(np.argmax(red_mask))
            if np.any(red_mask)
            else (
                int(np.argmax(np.abs(delta) > 1e-9))
                if np.any(np.abs(delta) > 1e-9)
                else 0
            )
        )

    # Split hourly profit Δ into energy-only piece (green/red) and payment (gold) stacked on top
    with_payment_hp = bid_hp.copy()
    payment_series = np.zeros(len(prices))
    if 0 <= pay_t < len(with_payment_hp):
        with_payment_hp[pay_t] += float(bid_cost)
        payment_series[pay_t] = float(bid_cost)

    energy_diff = bid_hp - base_hp
    colors = [("#2ca02c" if v >= 0 else "#d62728") for v in energy_diff]
    fig.add_trace(
        go.Bar(
            x=x, y=energy_diff, name="Hourly Profit Δ (energy)", marker_color=colors
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    if np.any(payment_series != 0):
        fig.add_trace(
            go.Bar(
                x=x, y=payment_series, name="Flexibility Payment", marker_color="gold"
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(base_hp),
            name="Baseline",
            legendgroup="baseline",
            showlegend=False,
            line=dict(color="black", dash="dash"),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(with_payment_hp),
            name="Flex Dispatched Cumulative Profit",
            line=dict(color="#ff7f0e", width=3),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=14),
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        margin=dict(t=80, r=30, l=70, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="$/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Grid Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (%)", row=3, col=1)
    fig.update_yaxes(title_text="Δ $/h", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative $", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Hour of Day", row=4, col=1)

    # Disabled file export while debugging; show in browser instead
    # fig.write_image(f"examples/bid_story_{tag}.pdf")
    fig.show()


def plot_generic_bid_story(
    prices,
    baseline_power_kw,
    bid_delta_p,
    agent_id,
    tag,
    bid_cost,
    event_len_hours,
    event_start_hour: Optional[int] = None,
):
    """
    Creates a clearer plot for a single consumption-only asset bid.
    """
    base_p = np.asarray(baseline_power_kw, dtype=float)
    delta = np.asarray(bid_delta_p, dtype=float)
    bid_p = base_p + delta

    # Cost calculations
    # Energy cost uses absolute consumption for loads
    base_cost = np.sum(prices * np.abs(base_p))
    bid_total_cost = np.sum(prices * np.abs(bid_p))
    net_savings = base_cost - bid_total_cost - bid_cost

    # 3-row layout: Price, Grid Power, Cost impact (bars + cumulative savings)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.22, 0.44, 0.34],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    x = list(range(len(prices)))
    # Power Profile in grid perspective for loads: grid = -abs(power)
    grid_base = -np.abs(base_p)
    grid_bid = -np.abs(bid_p)
    # Row 1: Price
    fig.add_trace(
        go.Scatter(x=x, y=prices, name="Price", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )

    # Row 2: Grid Power
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_base,
            name="Baseline",
            legendgroup="baseline",
            showlegend=True,
            line=dict(color="black", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_bid,
            name="Flex Dispatched Power",
            line=dict(color="#2ca02c", width=3),
        ),
        row=2,
        col=1,
    )

    # Highlight configured event window if provided; else auto-detect reduction
    n = len(prices)
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        s = max(0, int(event_start_hour))
        e = min(n - 1, s + int(event_len_hours) - 1)
        x0 = s - 0.5
        x1 = e + 0.5
        for rr in (1, 2, 3):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(255,165,0,0.12)",
                line_width=0,
                row=rr,
                col=1,
            )
    else:
        delta_grid = grid_bid - grid_base
        reduction_mask = delta_grid > 1e-9
        mask_idx = np.arange(len(delta_grid))
        reduction_mask = reduction_mask & (mask_idx >= 2)
        if np.any(reduction_mask):
            idx = np.where(reduction_mask)[0]
            starts = [idx[0]]
            ends = []
            for i in range(1, len(idx)):
                if idx[i] != idx[i - 1] + 1:
                    ends.append(idx[i - 1])
                    starts.append(idx[i])
            ends.append(idx[-1])
            for s, e in zip(starts, ends):
                x0 = s - 0.5
                x1 = e + 0.5
                for rr in (1, 2, 3):
                    fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor="rgba(255,165,0,0.12)",
                        line_width=0,
                        row=rr,
                        col=1,
                    )

    # Row 3: Profit — energy-only Δ (green/red) + golden payment bar stacked; cumulative lines on secondary axis
    base_hp = -prices * np.abs(base_p)
    new_hp = -prices * np.abs(bid_p)
    delta_grid = grid_bid - grid_base
    red_mask = delta_grid > 1e-9
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        pay_t = min(len(prices) - 1, int(event_start_hour) + int(event_len_hours))
    else:
        pay_t = (
            int(np.argmax(red_mask))
            if np.any(red_mask)
            else (
                int(np.argmax(np.abs(bid_p - base_p) > 1e-9))
                if np.any(np.abs(bid_p - base_p) > 1e-9)
                else 0
            )
        )
    with_payment_hp = new_hp.copy()
    payment_series = np.zeros(len(prices))
    if 0 <= pay_t < len(with_payment_hp):
        with_payment_hp[pay_t] += float(bid_cost)
        payment_series[pay_t] = float(bid_cost)
    energy_diff = new_hp - base_hp
    colors = [("#2ca02c" if v >= 0 else "#d62728") for v in energy_diff]
    fig.add_trace(
        go.Bar(
            x=x, y=energy_diff, name="Hourly Profit Δ (energy)", marker_color=colors
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    if np.any(payment_series != 0):
        fig.add_trace(
            go.Bar(
                x=x, y=payment_series, name="Flexibility Payment", marker_color="gold"
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(base_hp),
            name="Baseline",
            legendgroup="baseline",
            showlegend=False,
            line=dict(color="black", dash="dash"),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(with_payment_hp),
            name="Flex Dispatched Cumulative Profit",
            line=dict(color="#ff7f0e", width=3),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=14),
        barmode="relative",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        margin=dict(t=80, r=30, l=70, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(title_text="$/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Grid Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Δ $/h", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative $", row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Hour of Day", row=3, col=1)

    # Disabled file export while debugging; show in browser instead
    # fig.write_image(f"examples/bid_story_{tag}.pdf")
    fig.show()


def plot_hvac_temp_bid_story(
    prices: np.ndarray,
    baseline_power_kw: np.ndarray,
    bid_delta_p: np.ndarray,
    *,
    baseline_temp_c: np.ndarray,
    with_bid_temp_c: np.ndarray,
    setpoint_c: np.ndarray,
    agent_id: str,
    tag: str,
    bid_cost: float,
    event_len_hours: int,
    event_start_hour: Optional[int] = None,
):
    prices = np.asarray(prices, dtype=float)
    base_p = np.asarray(baseline_power_kw, dtype=float)
    delta = np.asarray(bid_delta_p, dtype=float)
    new_p = base_p + delta

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.18, 0.32, 0.26, 0.24],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )
    x = list(range(len(prices)))

    # Row 1: Price
    fig.add_trace(
        go.Scatter(x=x, y=prices, name="Price", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )

    # Row 2: Grid Power (load convention)
    grid_base = -np.abs(base_p)
    grid_new = -np.abs(new_p)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_base,
            name="Baseline Grid Power",
            line=dict(color="black", dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=grid_new,
            name="With-Bid Grid Power",
            line=dict(color="#2ca02c", width=3),
        ),
        row=2,
        col=1,
    )

    # Shade configured event window
    n = len(prices)
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        s = max(0, int(event_start_hour))
        e = min(n - 1, s + int(event_len_hours) - 1)
        x0 = s - 0.5
        x1 = e + 0.5
        for rr in (1, 2, 3, 4):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(255,165,0,0.12)",
                line_width=0,
                row=rr,
                col=1,
            )

    # Row 3: Temperature & Setpoint
    fig.add_trace(
        go.Scatter(
            x=x,
            y=baseline_temp_c,
            name="Baseline Temp (°C)",
            line=dict(color="black", dash="dash"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=with_bid_temp_c,
            name="With-Bid Temp (°C)",
            line=dict(color="#9467bd", width=3),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=setpoint_c,
            name="Setpoint (°C)",
            line=dict(color="#ff7f0e", dash="dot"),
        ),
        row=3,
        col=1,
    )

    # Row 4: Profit — energy-only Δ (green/red) + golden payment bar stacked; cumulative lines on secondary axis
    base_hp = -prices * np.abs(base_p)
    new_hp = -prices * np.abs(new_p)
    delta_grid = grid_new - grid_base
    red_mask = delta_grid > 1e-9
    if (
        event_start_hour is not None
        and event_len_hours is not None
        and event_len_hours > 0
    ):
        pay_t = min(len(prices) - 1, int(event_start_hour) + int(event_len_hours))
    else:
        pay_t = (
            int(np.argmax(red_mask))
            if np.any(red_mask)
            else (
                int(np.argmax(np.abs(delta) > 1e-9))
                if np.any(np.abs(delta) > 1e-9)
                else 0
            )
        )
    with_payment_hp = new_hp.copy()
    payment_series = np.zeros(len(prices))
    if 0 <= pay_t < len(with_payment_hp):
        with_payment_hp[pay_t] += float(bid_cost)
        payment_series[pay_t] = float(bid_cost)
    energy_diff = new_hp - base_hp
    colors = [("#2ca02c" if v >= 0 else "#d62728") for v in energy_diff]
    fig.add_trace(
        go.Bar(
            x=x, y=energy_diff, name="Hourly Profit Δ (energy)", marker_color=colors
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    if np.any(payment_series != 0):
        fig.add_trace(
            go.Bar(
                x=x, y=payment_series, name="Flexibility Payment", marker_color="gold"
            ),
            row=4,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(base_hp),
            name="Baseline Cum. Profit",
            line=dict(color="black", dash="dash"),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.cumsum(with_payment_hp),
            name="With-Bid Cum. Profit (incl. payment)",
            line=dict(color="#ff7f0e", width=3),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title=f"HVAC Bid Story — {agent_id} ({tag})",
        template="plotly_white",
        font=dict(size=14),
        barmode="relative",
    )
    fig.update_yaxes(title_text="$/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Grid Power (kW, negative = load)", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)
    fig.update_yaxes(title_text="Δ $/h", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="$ (cum.)", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Hour", row=4, col=1)

    fig.show()


def plot_baseline_with_bid_cloud(
    baseline_power_kw,
    bids: List[Dict],
    *,
    agent_id: str,
    tag: str,
    y_title: str = "Power (kW)",
):
    """
    Visualize an agent's baseline power schedule (gold line) alongside all
    alternative power profiles implied by its bids (light gray lines).

    - baseline_power_kw: array-like baseline power in the asset's convention
    - bids: list of bid dicts with 'delta_p' arrays (same length as baseline)
    - agent_id/tag: for title/context
    """
    base = np.asarray(baseline_power_kw, dtype=float)
    n = len(base)
    x = list(range(n))

    fig = go.Figure()

    # Alternative profiles in light gray
    first = True
    for b in bids or []:
        delta = np.asarray(b.get("delta_p", []), dtype=float)
        if len(delta) != n:
            continue
        alt = base + delta
        fig.add_trace(
            go.Scatter(
                x=x,
                y=alt,
                mode="lines",
                line=dict(color="rgba(128,128,128,0.35)", width=1),
                name="Alternative",
                showlegend=first,
            )
        )
        first = False

    # Baseline in gold
    fig.add_trace(
        go.Scatter(
            x=x,
            y=base,
            mode="lines",
            line=dict(color="#DAA520", width=3),
            name="Baseline",
        )
    )

    fig.update_layout(
        title=f"Baseline and Alternatives — {agent_id} ({tag})",
        template="plotly_white",
        font=dict(size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(t=70, r=30, l=70, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Hour of Day", showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.show()


# Keep other visualization functions from main file if needed
def plot_flexibility_supply_curve(
    bids: List[Dict],
    *,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
):
    """
    Aggregated flexibility supply curve for load reduction.
    X-axis: cumulative flexibility (kWh) within the selected window.
    Y-axis: marginal $/kWh using the bid's total kWh as the denominator.
    """
    if not bids:
        print("No flexibility bids available to plot.")
        return
    # Compute total reduction and $/kWh for each bid within optional window
    rows = []
    # Determine window mask
    # If indices not provided, include full horizon per bid length
    for b in bids:
        delta = np.asarray(b.get("delta_p", []), dtype=float)
        if delta.size == 0:
            continue
        n = delta.size
        lo = 0 if window_start is None else max(0, int(window_start))
        hi = (n - 1) if window_end is None else min(n - 1, int(window_end))
        if hi < lo:
            lo, hi = hi, lo
        mask = np.zeros(n, dtype=bool)
        mask[lo : hi + 1] = True
        reduction = -np.minimum(0.0, delta)  # positive where load is reduced
        tot_win = float(np.sum(reduction[mask]))
        tot_full = float(np.sum(reduction))
        cost = float(b.get("cost", np.inf))
        # Price using full-energy denominator to avoid inflating $/kWh when the
        # window captures only a small slice of the bid's energy.
        if tot_win > 0 and tot_full > 0 and np.isfinite(cost):
            rows.append((tot_win, cost / tot_full))
    if not rows:
        print("No bids with load reduction found.")
        return
    rows.sort(key=lambda x: x[1])
    cum = np.cumsum([r[0] for r in rows])
    y = [r[1] for r in rows]

    fig = go.Figure(
        go.Scatter(
            x=cum, y=y, mode="lines", line_shape="hv", fill="tozeroy", name="Supply"
        )
    )
    fig.update_layout(
        # no title for cleaner thesis figures
        xaxis_title="Cumulative Flexibility (kWh)",
        yaxis_title="Marginal Cost of Flexibility<br>($/kWh)",
        template="plotly_white",
        font=dict(size=16),
    )
    fig.show()


def plot_market_dispatch_results(
    baseline_load,
    target_load,
    achieved_load,
    dispatched_bids,
    *,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
):
    """
    Show baseline, target, achieved load; and contributions by agent type as stacked bars of reduction.
    """
    x = list(range(len(baseline_load)))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3]
    )

    # Baseline (dashed)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=baseline_load,
            name="System Baseline",
            line=dict(color="royalblue", dash="dash"),
        ),
        row=1,
        col=1,
    )
    # Load target: only visible within the requested window, if provided
    if (
        window_start is not None
        and window_end is not None
        and len(target_load) == len(x)
    ):
        n = len(target_load)
        s = max(0, int(window_start))
        e = min(n - 1, int(window_end))
        if e < s:
            s, e = e, s
        masked_target = [None] * n
        for i in range(s, e + 1):
            masked_target[i] = float(target_load[i])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=masked_target,
                name="Load Target",
                line=dict(color="firebrick"),
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=target_load,
                name="Load Target",
                line=dict(color="firebrick"),
            ),
            row=1,
            col=1,
        )
    # Achieved
    fig.add_trace(
        go.Scatter(
            x=x,
            y=achieved_load,
            name="Achieved Load",
            line=dict(color="black", width=3),
        ),
        row=1,
        col=1,
    )

    if dispatched_bids:
        # Stack by type
        from collections import defaultdict

        by_type = defaultdict(list)
        for b in dispatched_bids:
            by_type[b.get("agent_type", "Other")].append(
                np.asarray(b.get("delta_p", []), dtype=float)
            )
        colors = {
            "HVAC": "#FFA07A",
            "Storage": "#20B2AA",
            "Deferrable": "#9370DB",
            "C&I": "#4682B4",
        }
        for agent_type, deltas in by_type.items():
            total_delta = np.sum(np.stack(deltas), axis=0)
            reduction = -np.minimum(0.0, total_delta)  # positive reductions
            # Show bars only within the requested window; leave the rest empty
            if (
                window_start is not None
                and window_end is not None
                and len(reduction) == len(x)
            ):
                n = len(reduction)
                s = max(0, int(window_start))
                e = min(n - 1, int(window_end))
                if e < s:
                    s, e = e, s
                red_masked = [None] * n
                for i in range(s, e + 1):
                    red_masked[i] = float(reduction[i])
                y_vals = red_masked
            else:
                y_vals = reduction
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y_vals,
                    name=f"{agent_type} Contribution",
                    marker_color=colors.get(agent_type),
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        title_text="CBTFM Results: Peak Shaving Dispatch",
        template="plotly_white",
        font=dict(size=16),
        barmode="stack",
    )
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Dispatched Reduction (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Time of Day (Hour)", row=2, col=1)

    # Shade a specific window if provided (inclusive indices)
    if window_start is not None and window_end is not None and len(baseline_load) > 0:
        n = len(baseline_load)
        s = max(0, int(window_start))
        e = min(n - 1, int(window_end))
        if e < s:
            s, e = e, s
        x0 = s - 0.5
        x1 = e + 0.5
        for rr in (1, 2):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(255,165,0,0.12)",
                line_width=0,
                row=rr,
                col=1,
            )
    fig.show()


def plot_cost_percentile_stack(
    bids: List[Dict],
    n_timesteps: int,
    num_percentile_steps: int = 5,
    *,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
):
    """
    Total available reduction over time, stacked by cost percentile bands.

    Parameters
    ----------
    bids : list of bid dicts
        Each bid must include 'delta_p' and 'cost'.
    n_timesteps : int
        Number of hours to plot on the x-axis.
    num_percentile_steps : int, optional (default=5)
        Number of percentile bands to show (e.g., 5 → 0–20, 20–40, ..., 80–100).
    window_start : Optional[int]
        Start index (inclusive) of the time window to include in the stack. If None,
        the window starts at 0.
    window_end : Optional[int]
        End index (inclusive) of the time window to include in the stack. If None,
        the window ends at n_timesteps - 1.

    The band colors follow a yellow→red colormap (low→high cost). Each legend
    label includes the percentile range and the average $/kWh for that band.
    """
    if not bids:
        print("No bids available.")
        return
    rows = []
    for b in bids:
        delta = np.asarray(b.get("delta_p", []), dtype=float)
        reduction_profile = -np.minimum(0.0, delta)
        tot = float(np.sum(reduction_profile))
        if tot > 0 and np.isfinite(b.get("cost", np.inf)):
            rows.append((reduction_profile, float(b["cost"]) / tot))
    if not rows:
        print("No bids with load reduction found.")
        return

    profiles = np.stack([r[0] for r in rows])
    costs = np.array([r[1] for r in rows])

    # Build a mask for the window (inclusive indices)
    win_lo = 0 if window_start is None else max(0, int(window_start))
    win_hi = (
        (n_timesteps - 1)
        if window_end is None
        else min(n_timesteps - 1, int(window_end))
    )
    if win_hi < win_lo:
        win_lo, win_hi = win_hi, win_lo
    window_mask = np.zeros(n_timesteps, dtype=bool)
    window_mask[win_lo : win_hi + 1] = True

    # Build percentile edges (0..100) and corresponding cost quantile thresholds
    steps = max(1, int(num_percentile_steps))
    pct_edges = np.linspace(0, 100, steps + 1)
    if steps > 1:
        qs = np.quantile(costs, pct_edges[1:-1] / 100.0)
        bounds = np.concatenate(([-np.inf], qs, [np.inf]))
    else:
        bounds = np.array([-np.inf, np.inf])

    # Green→Yellow→Red gradient helper
    def _hex_gradient(n: int) -> List[str]:
        def _to_rgb(h: str):
            h = h.lstrip("#")
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

        def _blend(c1, c2, t: float):
            return tuple(int(round((1 - t) * a + t * b)) for a, b in zip(c1, c2))

        def _to_hex(rgb):
            return "#%02x%02x%02x" % rgb

        green = _to_rgb("#2ca02c")  # green
        yellow = _to_rgb("#ffe066")  # warm yellow
        red = _to_rgb("#bd0026")  # deep red
        if n <= 1:
            return ["#bd0026"]
        colors: List[str] = []
        for i in range(n):
            t = i / (n - 1)
            if t <= 0.5:
                # green → yellow
                tt = t / 0.5
                rgb = _blend(green, yellow, tt)
            else:
                # yellow → red
                tt = (t - 0.5) / 0.5
                rgb = _blend(yellow, red, tt)
            colors.append(_to_hex(rgb))
        return colors

    colors = _hex_gradient(steps)

    fig = go.Figure()
    bottom = np.zeros(n_timesteps)
    for i in range(steps):
        low_b, high_b = bounds[i], bounds[i + 1]
        if i < steps - 1:
            mask = (costs >= low_b) & (costs < high_b)
        else:
            mask = (costs >= low_b) & (costs <= high_b)
        if not np.any(mask):
            continue
        # Sum contributions only within the selected window
        band_full = np.zeros(n_timesteps, dtype=float)
        if np.any(mask):
            band_sum_win = np.sum(profiles[mask][:, window_mask], axis=0)
            band_full[window_mask] = band_sum_win
        avg_cost = float(np.mean(costs[mask])) if np.any(mask) else float("nan")
        label = f"{int(pct_edges[i])}–{int(pct_edges[i + 1])}% ($ {avg_cost:.2f}/kWh)"
        fig.add_trace(
            go.Scatter(
                x=list(range(n_timesteps)),
                y=bottom + band_full,
                fill="tonexty" if i > 0 else "tozeroy",
                mode="lines",
                name=label,
                line=dict(color=colors[i]),
            )
        )
        bottom += band_full

    fig.update_layout(
        title="Total Available Flexibility by Cost Percentile",
        xaxis_title="Time of Day (Hour)",
        yaxis_title="Dispatchable Flexibility (kW)",
        template="plotly_white",
        font=dict(size=16),
    )
    # Shade the selected window for clarity
    x0 = win_lo - 0.5
    x1 = win_hi + 0.5
    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,165,0,0.12)", line_width=0)
    # Show only hours 15–21 on the x-axis (inclusive)
    view_lo = max(0, 15)
    view_hi = min(n_timesteps - 1, 21)
    fig.update_xaxes(range=[view_lo - 0.5, view_hi + 0.5])

    fig.show()
