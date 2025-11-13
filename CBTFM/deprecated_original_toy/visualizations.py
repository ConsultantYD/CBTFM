import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_available_load(Delta_plus, costs):
    """
    Plot total available power showing the maximum potential contribution
    from all agents at each timestep, organized by cost efficiency.

    Parameters:
    -----------
    Delta_plus : np.array
        Array of shape (n_timesteps, n_agents, n_timesteps) containing possible load modifications
    costs : np.array
        Array of shape (n_agents, n_timesteps) containing costs per agent per timestep
    """
    plt.figure(figsize=(12, 8))

    # Create percentile bands for visualization
    percentiles = [0, 20, 40, 60, 80, 100]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(percentiles) - 1))

    n_timesteps, n_agents, _ = Delta_plus.shape

    # For each agent, find their largest total power offer
    optimal_contributions = []
    for agent in range(n_agents):
        max_total_power = 0
        best_profile = None
        best_cost = None

        for t in range(n_timesteps):
            profile = Delta_plus[t, agent]
            total_power = np.sum(profile)

            if total_power > max_total_power:
                max_total_power = total_power
                best_profile = profile
                best_cost = costs[agent, t]

        if best_profile is not None and max_total_power > 0:
            optimal_contributions.append((best_cost / max_total_power, best_profile))

    # Sort contributions by cost efficiency
    optimal_contributions.sort(key=lambda x: x[0])

    # Calculate power sums and costs for each percentile band
    bands = []
    cost_ranges = []
    for i in range(len(percentiles) - 1):
        start_idx = int(len(optimal_contributions) * percentiles[i] / 100)
        end_idx = int(len(optimal_contributions) * percentiles[i + 1] / 100)

        # Get profiles and costs in this percentile band
        band_data = optimal_contributions[start_idx:end_idx]
        if band_data:
            band_profiles = [prof for _, prof in band_data]
            band_costs = [cost for cost, _ in band_data]
            band_sum = np.sum(band_profiles, axis=0)
            bands.append(band_sum)
            cost_ranges.append((min(band_costs), max(band_costs)))

    # Plot percentile bands
    bottom = np.zeros(n_timesteps)
    for i, band in enumerate(bands):
        plt.fill_between(
            range(n_timesteps),
            bottom,
            bottom + band,
            color=colors[i],
            alpha=0.5,
            label=f"{percentiles[i]}-{percentiles[i+1]}th percentile\n({cost_ranges[i][0]:.2f}-{cost_ranges[i][1]:.2f} $/kW)",
        )
        bottom += band

    plt.title("Total Available Power by Cost Percentile")
    plt.xlabel("Time Steps")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Print some statistics
    print(f"Number of agents with valid profiles: {len(optimal_contributions)}")
    print(f"Maximum power available at final timestep: {bottom[-1]:.2f} kW")
    print(
        f"Average power per agent at final timestep: {bottom[-1]/len(optimal_contributions):.2f} kW"
    )
    min_cost = min(cost for cost, _ in optimal_contributions)
    max_cost = max(cost for cost, _ in optimal_contributions)
    print(f"Overall cost range: ${min_cost:.2f}/kW - ${max_cost:.2f}/kW")

    plt.show()
