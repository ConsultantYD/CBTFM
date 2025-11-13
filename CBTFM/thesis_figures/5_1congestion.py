import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pypsa

n = pypsa.examples.scigrid_de()

fig, ax = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.EqualEarth()},
)

load_distribution = n.loads_t.p_set.loc[n.snapshots[0]].groupby(n.loads.bus).sum()
n.plot(bus_size=load_distribution / 30000, ax=ax, title="Load distribution")
plt.show()

n.generators.groupby("carrier")["p_nom"].sum().round(1)
n.storage_units.groupby("carrier")["p_nom"].sum().round(1)

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
    ax.set_title(tech)
fig.tight_layout()
plt.show()

contingency_factor = 0.7
n.lines.s_max_pu = contingency_factor

n.lines.loc[["316", "527", "602"], "s_nom"] = 1715

n.optimize.optimize_with_rolling_horizon(horizon=4, overlap=0, log_to_console=False)
p_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
to_drop = p_by_carrier.max()[p_by_carrier.max() < 1700].index
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
p_by_carrier = p_by_carrier[cols]

c = [colors[col] for col in p_by_carrier.columns]
fig, ax = plt.subplots()
p_by_carrier.div(1e3).plot(kind="area", ax=ax, lw=0, color=c, alpha=0.7)
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02, 1, 0.2), frameon=False)
ax.set_ylabel("GW")
ax.set_xlabel("")
plt.show()

fig, ax = plt.subplots()

p_storage = n.storage_units_t.p.sum(axis=1)
state_of_charge = n.storage_units_t.state_of_charge.sum(axis=1)
p_storage.plot(label="Pumped hydro dispatch", ax=ax)
state_of_charge.plot(label="State of charge", ax=ax)

ax.axhline(0, color="k", lw=0.5, ls="--")
ax.legend()
ax.set_ylabel("MWh")
ax.set_xlabel("")

plt.show()


now = n.snapshots[4]
loading = n.lines_t.p0.loc[now] / n.lines.s_nom
loading.describe()

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EqualEarth()})
n.plot(
    ax=ax,
    line_color=loading.abs(),
    line_cmap="viridis",
    title="Line loading",
    bus_size=1e-3,
)

plt.show()

n.buses_t.marginal_price.loc[now].describe()
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

plt.hexbin(
    n.buses.x,
    n.buses.y,
    gridsize=20,
    C=n.buses_t.marginal_price.loc[now],
    cmap="viridis",
    zorder=-1,
)
n.plot(ax=ax, line_width=1, bus_size=0)

cb = plt.colorbar(location="right")
cb.set_label("Locational Marginal Price (€/MWh)")

plt.show()
