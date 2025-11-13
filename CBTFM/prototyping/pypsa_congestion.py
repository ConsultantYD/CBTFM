import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import pypsa

try:
    import mplcursors  # interactive hover tooltips
except Exception:
    mplcursors = None
_TOOLTIP_WARNED = False


def attach_bus_tooltips(ax, buses_df, *, transform):
    """Attach hover tooltips to a GeoAxes showing bus id/name.

    Adds an invisible scatter at bus coordinates so mplcursors can
    display annotations on hover. If mplcursors isn't installed, this
    is a no-op.
    """
    global _TOOLTIP_WARNED
    if mplcursors is None:
        if not _TOOLTIP_WARNED:
            print("Hover tooltips require 'mplcursors' (pip install mplcursors).")
            _TOOLTIP_WARNED = True
        return

    sc = ax.scatter(
        buses_df["x"].values,
        buses_df["y"].values,
        s=80,
        facecolors="none",
        edgecolors="none",
        alpha=0.0,
        zorder=10,
        transform=transform,
    )

    cursor = mplcursors.cursor(sc, hover=True)
    has_v_nom = "v_nom" in buses_df.columns

    @cursor.connect("add")
    def _on_add(sel):
        i = int(sel.index)
        bus_id = str(buses_df.index[i])
        if has_v_nom and pd.notna(buses_df["v_nom"].iloc[i]):
            text = f"{bus_id}\nV_nom: {buses_df['v_nom'].iloc[i]} kV"
        else:
            text = bus_id
        sel.annotation.set_text(text)
        try:
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
        except Exception:
            pass


def label_buses_on_map(ax, buses_df, *, transform):
    """Draw small text labels for each bus (by index) on the map."""
    names = buses_df.index.astype(str).tolist()
    xs = buses_df["x"].values
    ys = buses_df["y"].values
    for name, x, y in zip(names, xs, ys):
        txt = ax.text(
            x,
            y,
            name,
            fontsize=6,
            ha="left",
            va="bottom",
            zorder=11,
            transform=transform,
        )
        try:
            txt.set_path_effects(
                [pe.withStroke(linewidth=2, foreground="white", alpha=0.9)]
            )
        except Exception:
            pass


n = pypsa.examples.scigrid_de()

# (Plots removed to keep this file focused on congestion diagnostics)


print(n.generators.groupby("carrier")["p_nom"].sum().round(1))
print(n.storage_units.groupby("carrier")["p_nom"].sum().round(1))

# (Technology distribution subplots removed)

contingency_factor = 0.7
n.lines.s_max_pu = contingency_factor
n.lines.loc[["316", "527", "602"], "s_nom"] = 1715

n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=0, log_to_console=False)

# =============================
# Focused congestion diagnostics
# =============================


# Resolve target buses dynamically: 182, 180, and a 220kV neighbor of 180
def _resolve_bus_label(n, base_id: str):
    base_id = str(base_id)
    idx = n.buses.index.astype(str)
    # Prefer exact match, else any with prefix 'base_id_' (e.g., '180_220kV')
    exact = [b for b in idx if b == base_id]
    if exact:
        return exact[0]
    prefixed = [b for b in idx if b.split("_")[0] == base_id]
    # If multiple, prefer 220kV if present
    for kv in ("220kV", "380kV", "110kV"):
        kv_matches = [b for b in prefixed if kv in b]
        if kv_matches:
            return kv_matches[0]
    return prefixed[0] if prefixed else None


bus_182_label = _resolve_bus_label(n, "182")
bus_180_label = _resolve_bus_label(n, "180")

# Find a 220kV neighbor connected to bus_180_label
neighbor_220_label = None
try:
    if bus_180_label is not None:
        lines_df = n.lines
        mask_180 = (lines_df.bus0.astype(str) == bus_180_label) | (
            lines_df.bus1.astype(str) == bus_180_label
        )
        neighs = []
        for lid in lines_df.index[mask_180]:
            b0 = str(lines_df.loc[lid, "bus0"])
            b1 = str(lines_df.loc[lid, "bus1"])
            neighs.append(b1 if b0 == bus_180_label else b0)
        # Prefer neighbors with 220kV in the label
        cand_220 = [b for b in neighs if "220kV" in str(b)]
        if cand_220:
            neighbor_220_label = cand_220[0]
        elif neighs:
            neighbor_220_label = neighs[0]
except Exception:
    neighbor_220_label = None

TARGET_BUSES = [
    b for b in [bus_182_label, bus_180_label, neighbor_220_label] if b is not None
]
if len(TARGET_BUSES) < 3:
    print("Warning: could not resolve all requested buses (182, 180, neighbor_220kV).")

try:
    # Aggregate time series to bus level
    loads_by_bus = n.loads_t.p_set.groupby(n.loads.bus, axis=1).sum()
    gens_by_bus = n.generators_t.p.groupby(n.generators.bus, axis=1).sum()

    # Use same time formatting as earlier area plot if available
    try:
        _xlim = (tmin, tmax)
        _loc = hour_locator
        _fmt = hour_formatter
    except NameError:
        # Fallback to auto locators if earlier variables are not defined
        _xlim = (n.snapshots.min(), n.snapshots.max())
        _loc = mdates.AutoDateLocator(minticks=5, maxticks=8)
        _fmt = mdates.ConciseDateFormatter(_loc)

    for bus in TARGET_BUSES:
        load_series = (
            loads_by_bus[bus]
            if bus in loads_by_bus.columns
            else pd.Series(0.0, index=n.snapshots)
        )
        gen_series = (
            gens_by_bus[bus]
            if bus in gens_by_bus.columns
            else pd.Series(0.0, index=n.snapshots)
        )
        # Ensure alignment and dtype
        load_series = load_series.astype(float).reindex(n.snapshots).sort_index()
        gen_series = gen_series.astype(float).reindex(n.snapshots).sort_index()

        fig_lg, ax_lg = plt.subplots(figsize=(12, 4))
        ax_lg.plot(
            load_series.index,
            load_series.values,
            label="Load [MW]",
            color="tab:red",
            lw=2,
            alpha=0.9,
        )
        ax_lg.plot(
            gen_series.index,
            gen_series.values,
            label="Generation [MW]",
            color="tab:green",
            lw=2,
            alpha=0.9,
        )
        ax_lg.set_xlim(*_xlim)
        ax_lg.xaxis.set_major_locator(_loc)
        ax_lg.xaxis.set_major_formatter(_fmt)
        ax_lg.set_ylabel("MW")
        ax_lg.set_xlabel("Time")
        ax_lg.set_title(f"Load vs Generation at bus {bus}")
        ax_lg.legend(loc="upper left", frameon=False)
        ax_lg.grid(True, alpha=0.2)
        fig_lg.tight_layout()
        plt.show()
except Exception as e:
    print(f"Failed to plot load/generation per bus: {e}")

# Line loading over time: 180 <-> neighbor(220kV)
BUS_A = bus_180_label if bus_180_label is not None else "180"
BUS_B = neighbor_220_label if neighbor_220_label is not None else ""
try:
    # Find all lines connecting BUS_A and BUS_B (both directions)
    lines_df = n.lines
    mask = (
        (lines_df.bus0.astype(str) == BUS_A) & (lines_df.bus1.astype(str) == BUS_B)
    ) | ((lines_df.bus0.astype(str) == BUS_B) & (lines_df.bus1.astype(str) == BUS_A))
    line_ids = lines_df.index[mask]

    fig_line, ax_line = plt.subplots(figsize=(12, 4))
    if len(line_ids) == 0:
        ax_line.text(
            0.5,
            0.5,
            f"No line found between {BUS_A} and {BUS_B}",
            ha="center",
            transform=ax_line.transAxes,
        )
        ax_line.axis("off")
    else:
        # Plot per-line utilization; binding at 1.0 (i.e., |p| == s_max_pu * s_nom)
        for lid in line_ids:
            p0 = n.lines_t.p0[lid].astype(float)
            cap = float(lines_df.loc[lid, "s_nom"]) * float(
                lines_df.loc[lid, "s_max_pu"]
            )
            util = p0.abs() / cap if cap > 0 else p0.abs()
            util = util.reindex(n.snapshots).sort_index()
            ax_line.plot(util.index, util.values, lw=2, label=f"Line {lid}")

        # Binding threshold
        ax_line.axhline(1.0, color="gray", ls="--", lw=1.5, label="Binding threshold")
        # Time axis formatting
        try:
            ax_line.set_xlim(*_xlim)
            ax_line.xaxis.set_major_locator(_loc)
            ax_line.xaxis.set_major_formatter(_fmt)
        except Exception:
            pass
        ax_line.set_ylim(0, None)
        ax_line.set_ylabel("Utilization (|p| / (s_max_pu*s_nom))")
        ax_line.set_xlabel("Time")
        ax_line.set_title(f"Line loading between {BUS_A} and {BUS_B}")
        ax_line.legend(loc="upper left", frameon=False)
        ax_line.grid(True, alpha=0.2)
    fig_line.tight_layout()
    plt.show()
except Exception as e:
    print(f"Failed to plot line loading {BUS_A}<->{BUS_B}: {e}")

## (Removed: previous generation-cap loop to keep file focused on source/sink shift)

## Restored: Scenario to cap generation at bus 182 during 0–12h and assess line loading
try:
    TARGET_GEN_BUS = bus_182_label if "bus_182_label" in globals() else None
    if TARGET_GEN_BUS is None:
        print("Could not resolve bus 182 label; skipping generation-cap scenarios.")
    else:
        # Lines between A and B (as above)
        lines_df = n.lines
        mask = (
            (lines_df.bus0.astype(str) == BUS_A) & (lines_df.bus1.astype(str) == BUS_B)
        ) | (
            (lines_df.bus0.astype(str) == BUS_B) & (lines_df.bus1.astype(str) == BUS_A)
        )
        line_ids = lines_df.index[mask]

        if len(line_ids) == 0:
            print(
                f"No line found between {BUS_A} and {BUS_B} for the generation-cap scenarios."
            )
        else:
            # Ensure we have a p_max_pu table to edit
            baseline_p_max = getattr(n.generators_t, "p_max_pu", None)
            if baseline_p_max is None or baseline_p_max.empty:
                baseline_p_max = pd.DataFrame(
                    1.0, index=n.snapshots, columns=n.generators.index
                )
            else:
                baseline_p_max = baseline_p_max.copy()

            # Prepare baseline utilization
            utils_base = []
            for lid in line_ids:
                p0 = n.lines_t.p0[lid].astype(float)
                cap = float(n.lines.loc[lid, "s_nom"]) * float(
                    n.lines.loc[lid, "s_max_pu"]
                )
                util = p0.abs() / cap if cap > 0 else p0.abs()
                utils_base.append(util.reindex(n.snapshots).sort_index())
            util_base = pd.concat(utils_base, axis=1).max(axis=1)

            # Generators at bus 182
            gens_idx_target = n.generators.index[
                n.generators.bus.astype(str) == TARGET_GEN_BUS
            ]
            total_p_nom = float(n.generators.loc[gens_idx_target, "p_nom"].sum())
            if total_p_nom <= 0 or len(gens_idx_target) == 0:
                print(
                    f"No generators found at bus {TARGET_GEN_BUS}; scenarios will have no effect."
                )

            # Time mask: hours 0..12 inclusive
            idx = baseline_p_max.index
            mask_0_12 = (idx.hour >= 0) & (idx.hour <= 12)

            GEN_LEVELS_MW = [0, 5, 10, 20, 30, 40, 50]

            fig_comp, ax_comp = plt.subplots(figsize=(12, 4))
            # Use hour-only formatting
            try:
                _xlim = (_xlim[0], _xlim[1])
                _loc = _loc
                _fmt = _fmt
            except Exception:
                tmin2, tmax2 = n.snapshots.min(), n.snapshots.max()
                total_hours2 = max(
                    1,
                    int(
                        np.ceil(
                            (pd.to_datetime(tmax2) - pd.to_datetime(tmin2))
                            / pd.Timedelta(hours=1)
                        )
                    ),
                )
                hour_interval2 = max(1, int(np.ceil(total_hours2 / 8)))
                _xlim = (tmin2, tmax2)
                _loc = mdates.HourLocator(interval=hour_interval2)
                _fmt = mdates.DateFormatter("%H:%M")

            # Plot baseline
            ax_comp.plot(
                util_base.index,
                util_base.values,
                lw=2.5,
                color="black",
                label="Baseline",
            )

            summary = []
            for mw in GEN_LEVELS_MW:
                try:
                    # Reset and apply cap ratio
                    n.generators_t.p_max_pu = baseline_p_max.copy()
                    r = 0.0 if total_p_nom <= 0 else mw / total_p_nom
                    r = float(np.clip(r, 0.0, 1.0))
                    # Cap each target generator identically so sum cap ~ mw
                    base_slice = n.generators_t.p_max_pu.loc[mask_0_12, gens_idx_target]
                    n.generators_t.p_max_pu.loc[mask_0_12, gens_idx_target] = (
                        np.minimum(base_slice.to_numpy(), r)
                    )

                    # Re-optimize
                    n.optimize.optimize_with_rolling_horizon(
                        horizon=4, overlap=0, log_to_console=False
                    )

                    # Compute aggregated utilization across found lines
                    utils_new = []
                    for lid in line_ids:
                        p0 = n.lines_t.p0[lid].astype(float)
                        cap = float(n.lines.loc[lid, "s_nom"]) * float(
                            n.lines.loc[lid, "s_max_pu"]
                        )
                        util = p0.abs() / cap if cap > 0 else p0.abs()
                        utils_new.append(util.reindex(n.snapshots).sort_index())
                    util_new = pd.concat(utils_new, axis=1).max(axis=1)

                    ax_comp.plot(
                        util_new.index,
                        util_new.values,
                        lw=1.8,
                        label=f"cap {mw} MW (0–12h)",
                    )
                    summary.append((mw, float(util_new.max())))
                except Exception as _e:
                    print(f"Failed generation-cap scenario {mw} MW: {_e}")

            # Formatting
            ax_comp.axhline(
                1.0, color="gray", ls="--", lw=1.5, label="Binding threshold"
            )
            ax_comp.set_xlim(*_xlim)
            ax_comp.xaxis.set_major_locator(_loc)
            ax_comp.xaxis.set_major_formatter(_fmt)
            ax_comp.set_ylim(0, None)
            ax_comp.set_ylabel("Utilization (|p| / (s_max_pu*s_nom))")
            ax_comp.set_xlabel("Time")
            ax_comp.set_title(
                f"Effect of capping gen at {TARGET_GEN_BUS} (0–12h) on {BUS_A}<->{BUS_B}"
            )
            ax_comp.legend(loc="upper left", frameon=False, ncol=3)
            ax_comp.grid(True, alpha=0.2)
            fig_comp.tight_layout()
            plt.show()

            # Summary table
            print("\nGeneration-cap scenarios at bus", TARGET_GEN_BUS)
            for mw, m in summary:
                status = "OK (<=1.0)" if m <= 1.0 + 1e-9 else "Binding (>1.0)"
                print(f"  gen 0–12h capped at {mw:.1f} MW -> max_util={m:.3f} {status}")

            # Restore baseline generator limits
            n.generators_t.p_max_pu = baseline_p_max
except Exception as e:
    print(f"Failed generation-cap scenarios: {e}")

# =============================
# Load-shift experiment (A up, B down by %)
# =============================
# Increase load at BUS_A by +pct% while decreasing at BUS_B by -pct%.
# Re-optimize and plot resulting line utilization over time for each pct.

try:
    # Reuse BUS_A/B defined earlier (BUS_A = 180 label, BUS_B = 220kV neighbor)
    lines_df = n.lines
    mask = (
        (lines_df.bus0.astype(str) == BUS_A) & (lines_df.bus1.astype(str) == BUS_B)
    ) | ((lines_df.bus0.astype(str) == BUS_B) & (lines_df.bus1.astype(str) == BUS_A))
    line_ids = lines_df.index[mask]

    if len(line_ids) == 0:
        print(
            f"No line found between {BUS_A} and {BUS_B} for the load-shift experiment."
        )
    else:
        PCT_STEPS = list(range(0, 91, 10))  # 0%, 10%, ..., 90%
        baseline_p_set = n.loads_t.p_set.copy()

        # Prepare time axis formatting (hours only)
        try:
            _xlim = (_xlim[0], _xlim[1])  # from earlier block if available
            _loc = _loc
            _fmt = _fmt
        except Exception:
            tmin2, tmax2 = n.snapshots.min(), n.snapshots.max()
            total_hours2 = max(
                1,
                int(
                    np.ceil(
                        (pd.to_datetime(tmax2) - pd.to_datetime(tmin2))
                        / pd.Timedelta(hours=1)
                    )
                ),
            )
            hour_interval2 = max(1, int(np.ceil(total_hours2 / 8)))
            _xlim = (tmin2, tmax2)
            _loc = mdates.HourLocator(interval=hour_interval2)
            _fmt = mdates.DateFormatter("%H:%M")

        # Identify load indices on each bus
        loads_idx_A = n.loads.index[n.loads.bus.astype(str) == BUS_A]
        loads_idx_B = n.loads.index[n.loads.bus.astype(str) == BUS_B]

        fig_shift, ax_shift = plt.subplots(figsize=(12, 4))
        summary = []

        # Plot baseline utilization first
        utils_base = []
        for lid in line_ids:
            p0 = n.lines_t.p0[lid].astype(float)
            cap = float(n.lines.loc[lid, "s_nom"]) * float(n.lines.loc[lid, "s_max_pu"])
            util = p0.abs() / cap if cap > 0 else p0.abs()
            utils_base.append(util.reindex(n.snapshots).sort_index())
        util_base = pd.concat(utils_base, axis=1).max(axis=1)
        ax_shift.plot(
            util_base.index, util_base.values, lw=2.5, color="black", label="Baseline"
        )

        for pct in PCT_STEPS:
            try:
                factor_a = 1.0 + pct / 100.0
                factor_b = max(0.0, 1.0 - pct / 100.0)

                # Apply scaled loads to a fresh copy of baseline
                n.loads_t.p_set.loc[:, loads_idx_A] = (
                    baseline_p_set[loads_idx_A] * factor_a
                )
                n.loads_t.p_set.loc[:, loads_idx_B] = (
                    baseline_p_set[loads_idx_B] * factor_b
                )

                # Re-optimize dispatch
                n.optimize.optimize_with_rolling_horizon(
                    horizon=4, overlap=0, log_to_console=False
                )

                # Aggregate utilization across parallel lines (max across ids)
                utils = []
                for lid in line_ids:
                    p0 = n.lines_t.p0[lid].astype(float)
                    cap = float(n.lines.loc[lid, "s_nom"]) * float(
                        n.lines.loc[lid, "s_max_pu"]
                    )
                    util = p0.abs() / cap if cap > 0 else p0.abs()
                    utils.append(util.reindex(n.snapshots).sort_index())
                util_agg = pd.concat(utils, axis=1).max(axis=1)

                ax_shift.plot(
                    util_agg.index,
                    util_agg.values,
                    lw=2,
                    label=f"+{pct}% at {BUS_A}, -{pct}% at {BUS_B}",
                    alpha=0.9,
                )
                summary.append((pct, float(util_agg.max())))
            except Exception as _e:
                print(f"Failed scenario {pct}%: {_e}")

        # Format axis and add binding line
        ax_shift.axhline(1.0, color="gray", ls="--", lw=1.5, label="Binding threshold")
        ax_shift.set_xlim(*_xlim)
        ax_shift.xaxis.set_major_locator(_loc)
        ax_shift.xaxis.set_major_formatter(_fmt)
        ax_shift.set_ylim(0, None)
        ax_shift.set_ylabel("Utilization (|p| / (s_max_pu*s_nom))")
        ax_shift.set_xlabel("Time")
        ax_shift.set_title(f"Effect of load shift on {BUS_A}<->{BUS_B} utilization")
        ax_shift.legend(loc="upper left", frameon=False, ncol=2)
        ax_shift.grid(True, alpha=0.2)
        fig_shift.tight_layout()
        plt.show()

        # Print concise summary table
        if summary:
            print("\nLoad-shift summary (max utilization across the horizon):")
            for pct, m in summary:
                status = "OK (<=1.0)" if m <= 1.0 + 1e-9 else "Binding (>1.0)"
                print(
                    f"  pct=+{pct}% on {BUS_A}, -{pct}% on {BUS_B}: max_util={m:.3f} -> {status}"
                )

        # Restore baseline loads
        n.loads_t.p_set = baseline_p_set
except Exception as e:
    print(f"Failed load-shift experiment: {e}")

# =============================
# Delta utilization: cap gen at bus 182 to 5 MW (05–09h) vs baseline
# =============================
try:
    TARGET_GEN_BUS = bus_182_label if "bus_182_label" in globals() else None
    if not BUS_A or not BUS_B or TARGET_GEN_BUS is None:
        print("Cannot run 5MW cap delta: BUS_A/B or bus 182 not resolved.")
    else:
        # Identify the A<->B lines
        lines_df = n.lines
        mask = (
            (lines_df.bus0.astype(str) == BUS_A) & (lines_df.bus1.astype(str) == BUS_B)
        ) | (
            (lines_df.bus0.astype(str) == BUS_B) & (lines_df.bus1.astype(str) == BUS_A)
        )
        line_ids = lines_df.index[mask]
        if len(line_ids) == 0:
            print(f"No line found between {BUS_A} and {BUS_B} for the 5MW cap delta.")
        else:
            # Preserve baseline settings
            baseline_p_set = n.loads_t.p_set.copy()
            base_p_max = getattr(n.generators_t, "p_max_pu", None)
            if base_p_max is None or base_p_max.empty:
                base_p_max = pd.DataFrame(
                    1.0, index=n.snapshots, columns=n.generators.index
                )
            else:
                base_p_max = base_p_max.copy()

            # Re-solve baseline to ensure a fresh reference
            n.optimize.optimize_with_rolling_horizon(
                horizon=4, overlap=0, log_to_console=False
            )

            # Baseline utilization
            utils_base = []
            for lid in line_ids:
                p0 = n.lines_t.p0[lid].astype(float)
                cap = float(n.lines.loc[lid, "s_nom"]) * float(
                    n.lines.loc[lid, "s_max_pu"]
                )
                util = p0.abs() / cap if cap > 0 else p0.abs()
                utils_base.append(util.reindex(n.snapshots).sort_index())
            util_base = pd.concat(utils_base, axis=1).max(axis=1)

            # Apply 5 MW cap at bus 182 for 05–09h
            gens_idx = n.generators.index[
                n.generators.bus.astype(str) == TARGET_GEN_BUS
            ]
            total_p_nom = float(n.generators.loc[gens_idx, "p_nom"].sum())
            if total_p_nom <= 0 or len(gens_idx) == 0:
                print(
                    f"No generators at bus {TARGET_GEN_BUS}; delta scenario has no effect."
                )
            else:
                # Build masked cap table
                idx = n.snapshots
                mask_5_9 = (idx.hour >= 5) & (idx.hour <= 9)
                n.generators_t.p_max_pu = base_p_max.copy()
                r = float(np.clip(5.0 / total_p_nom, 0.0, 1.0))
                base_slice = n.generators_t.p_max_pu.loc[mask_5_9, gens_idx]
                n.generators_t.p_max_pu.loc[mask_5_9, gens_idx] = np.minimum(
                    base_slice.to_numpy(), r
                )

                # Re-solve capped scenario
                n.optimize.optimize_with_rolling_horizon(
                    horizon=4, overlap=0, log_to_console=False
                )

                # New utilization
                utils_new = []
                for lid in line_ids:
                    p0 = n.lines_t.p0[lid].astype(float)
                    cap = float(n.lines.loc[lid, "s_nom"]) * float(
                        n.lines.loc[lid, "s_max_pu"]
                    )
                    util = p0.abs() / cap if cap > 0 else p0.abs()
                    utils_new.append(util.reindex(n.snapshots).sort_index())
                util_new = pd.concat(utils_new, axis=1).max(axis=1)

                # Delta visualization (capped - baseline)
                delta = (util_new - util_base).astype(float)

                fig_delta, ax_delta = plt.subplots(figsize=(12, 4))
                ax_delta.plot(
                    delta.index,
                    delta.values,
                    color="tab:blue",
                    lw=2,
                    label="Δ utilization (cap - base)",
                )
                # Fill positive (red) and negative (green) areas
                ax_delta.fill_between(
                    delta.index,
                    0,
                    delta.values,
                    where=(delta.values >= 0),
                    color="tab:red",
                    alpha=0.18,
                    label="Higher vs base",
                )
                ax_delta.fill_between(
                    delta.index,
                    0,
                    delta.values,
                    where=(delta.values < 0),
                    color="tab:green",
                    alpha=0.18,
                    label="Lower vs base",
                )
                # Shade the cap window
                try:
                    t_start = delta.index[(delta.index.hour >= 5)][0]
                    t_end_candidates = delta.index[(delta.index.hour <= 9)]
                    t_end = t_end_candidates[-1] if len(t_end_candidates) else t_start
                    ax_delta.axvspan(
                        t_start,
                        t_end,
                        color="gray",
                        alpha=0.1,
                        label="Cap window (05–09h)",
                    )
                except Exception:
                    pass

                # Formatting
                try:
                    ax_delta.set_xlim(*_xlim)
                    ax_delta.xaxis.set_major_locator(_loc)
                    ax_delta.xaxis.set_major_formatter(_fmt)
                except Exception:
                    pass
                ax_delta.axhline(0.0, color="black", lw=1)
                ax_delta.set_ylabel("Δ Utilization")
                ax_delta.set_xlabel("Time")
                ax_delta.set_title(
                    f"Δ utilization on {BUS_A}<->{BUS_B}: cap 5 MW at {TARGET_GEN_BUS} (05–09h)"
                )
                ax_delta.legend(loc="upper left", frameon=False, ncol=2)
                ax_delta.grid(True, alpha=0.2)
                fig_delta.tight_layout()
                plt.show()

                # Quick summary
                print("\nDelta summary (cap 5 MW at", TARGET_GEN_BUS, "05–09h):")
                print(
                    "  max increase:",
                    f"{delta.max():.3f}",
                    "  max decrease:",
                    f"{delta.min():.3f}",
                )

            # Restore baseline state
            n.generators_t.p_max_pu = base_p_max
            n.loads_t.p_set = baseline_p_set
            # Optionally re-optimize to reset to base
            n.optimize.optimize_with_rolling_horizon(
                horizon=4, overlap=0, log_to_console=False
            )
except Exception as e:
    print(f"Failed 5MW cap delta scenario: {e}")
