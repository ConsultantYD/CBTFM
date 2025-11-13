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

"""
https://docs.pypsa.org/latest/examples/scigrid-lopf-then-pf/#locational-marginal-prices
"""


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

fig, ax = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.EqualEarth()},
)

load_distribution = n.loads_t.p_set.loc[n.snapshots[0]].groupby(n.loads.bus).sum()
n.plot(bus_size=load_distribution / 30000, ax=ax, title="Load distribution")
attach_bus_tooltips(ax, n.buses, transform=ccrs.PlateCarree())
# label_buses_on_map(ax, n.buses, transform=ccrs.PlateCarree())  # commented to declutter
plt.show()


print(n.generators.groupby("carrier")["p_nom"].sum().round(1))
print(n.storage_units.groupby("carrier")["p_nom"].sum().round(1))

techs = ["Gas", "Brown Coal", "Hard Coal", "Wind Offshore", "Wind Onshore", "Solar"]

n_graphs = len(techs)
n_cols = 3
if n_graphs % n_cols == 0:
    n_rows = n_graphs // n_cols
else:
    n_rows = n_graphs // n_cols + 1


fig, axes = plt.subplots(
    nrows=n_rows, ncols=n_cols, subplot_kw={"projection": ccrs.EqualEarth()}
)
size = 6
fig.set_size_inches(size * n_cols, size * n_rows)

for i, tech in enumerate(techs):
    i_row = i // n_cols
    i_col = i % n_cols

    ax = axes[i_row, i_col]
    gens = n.generators[n.generators.carrier == tech]
    gen_distribution = (
        gens.groupby("bus").sum()["p_nom"].reindex(n.buses.index, fill_value=0)
    )
    n.plot(ax=ax, bus_size=gen_distribution / 20000)
    attach_bus_tooltips(ax, n.buses, transform=ccrs.PlateCarree())
    ax.set_title(tech)
fig.tight_layout()
plt.show()

contingency_factor = 0.7
n.lines.s_max_pu = contingency_factor
n.lines.loc[["316", "527", "602"], "s_nom"] = 1715

n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=0, log_to_console=False)

p_by_carrier_full = n.generators_t.p.T.groupby(n.generators.carrier).sum().T

# Keep a filtered copy for the stacked-area plot, but make sure we don't
# accidentally drop 'Gas' (so the bottom subplot can still show it)
p_by_carrier = p_by_carrier_full.copy()
_max_series = p_by_carrier.max()
to_drop = [col for col, val in _max_series.items() if (val < 1700 and col != "Gas")]
p_by_carrier.drop(to_drop, axis=1, inplace=True)
p_by_carrier.columns

colors = {
    "Brown Coal": "brown",
    "Hard Coal": "k",
    "Nuclear": "r",
    "Run of River": "green",
    "Wind Onshore": "blue",
    "Solar": "yellow",
    "Wind Offshore": "cyan",
    "Waste": "orange",
    "Gas": "orange",
}
# reorder
cols = [
    "Nuclear",
    "Run of River",
    "Brown Coal",
    "Hard Coal",
    "Gas",
    "Wind Offshore",
    "Wind Onshore",
    "Solar",
]
# Only keep columns that actually exist to avoid KeyError
p_by_carrier = p_by_carrier[[col for col in cols if col in p_by_carrier.columns]]

c = [colors[col] for col in p_by_carrier.columns]
p_by_carrier_plot = p_by_carrier.copy()

# Prepare a consistent time axis for subsequent figures (hours only)
time_index = p_by_carrier_plot.index.sort_values()
tmin, tmax = time_index.min(), time_index.max()
# Aim for ~8 ticks across the range, using hourly locator
total_hours = max(
    1,
    int(np.ceil((pd.to_datetime(tmax) - pd.to_datetime(tmin)) / pd.Timedelta(hours=1))),
)
hour_interval = max(1, int(np.ceil(total_hours / 8)))
hour_locator = mdates.HourLocator(interval=hour_interval)
hour_formatter = mdates.DateFormatter("%H:%M")

# Figure 1 (time series): Stacked area plot by carrier (GW)
fig, ax = plt.subplots()
p_by_carrier.div(1e3).plot(kind="area", ax=ax, lw=0, color=c, alpha=0.7)
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02, 1, 0.2), frameon=False)
ax.set_ylabel("GW")
ax.set_xlabel("")
plt.show()

# focused plot of Gas output at a specific bus (node 395).
TARGET_BUS = "395"
# Zero out values before 08:00 to focus on afternoon
_mask_before8 = p_by_carrier.index.hour < 8
p_by_carrier_plot = p_by_carrier.copy()
p_by_carrier_plot.loc[_mask_before8, :] = 0

try:
    gens = n.generators.copy()
    # Match common gas carriers (e.g., 'Gas', 'OCGT', 'CCGT')
    gas_mask = gens.carrier.astype(str).str.contains(
        r"\b(gas|ocgt|ccgt)\b", case=False, regex=True, na=False
    )
    bus_mask = gens.bus.astype(str) == str(TARGET_BUS)
    gas_gens_at_bus = gens.index[gas_mask & bus_mask]

    fig_gas, ax_gas = plt.subplots(figsize=(12, 4))
    if len(gas_gens_at_bus) == 0:
        ax_gas.text(
            0.5,
            0.5,
            f"No gas generator found at bus {TARGET_BUS}",
            ha="center",
            transform=ax_gas.transAxes,
        )
        ax_gas.axis("off")
    else:
        series = (
            n.generators_t.p[gas_gens_at_bus].sum(axis=1).astype(float).sort_index()
        )
        # Zero out hours before 08:00
        series_plot = series.copy()
        series_plot.loc[series_plot.index.hour < 8] = 0.0
        ax_gas.plot(
            series_plot.index,
            series_plot.values,
            color="orange",
            lw=2,
            label=f"Gas at bus {TARGET_BUS}",
        )
        ax_gas.fill_between(
            series_plot.index, 0, series_plot.values, color="orange", alpha=0.15
        )
        peak_time = series_plot.idxmax()
        peak_val = float(series_plot.max())
        ax_gas.scatter([peak_time], [peak_val], color="orange", zorder=5)
        ax_gas.annotate(
            f"Peak: {peak_val:,.0f} MW",
            xy=(peak_time, peak_val),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )
        ax_gas.set_ylabel("MW")
        ax_gas.set_xlabel("Time")
        ax_gas.set_xlim(tmin, tmax)
        ax_gas.xaxis.set_major_locator(hour_locator)
        ax_gas.xaxis.set_major_formatter(hour_formatter)
        # ax_gas.set_title(f"Gas output at bus {TARGET_BUS}")
        ax_gas.legend(loc="upper left", frameon=False)
        ax_gas.grid(True, alpha=0.2)
    fig_gas.tight_layout()
    plt.show()
    # Print the gas time series (MW) for reuse (CSV format)
    try:
        gas_series_name = f"Gas_bus_{TARGET_BUS}_MW"
        gas_csv = series_plot.rename(gas_series_name).to_csv()
        print("\nTime series (CSV) for natural gas at bus " + TARGET_BUS + ":\n")
        print(gas_csv)
    except Exception as _:
        pass
except Exception as e:
    print(f"Failed to plot Gas at bus {TARGET_BUS}: {e}")
