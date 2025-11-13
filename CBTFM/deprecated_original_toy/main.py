# main.py
import numpy as np
import matplotlib.pyplot as plt
from visualizations import plot_available_load  # plot_executive_view


def generate_realistic_submissions(n_submissions, n_timesteps):
    delta_plus = np.zeros((n_submissions, n_timesteps))
    for i in range(n_submissions):
        start_time = np.random.randint(0, n_timesteps - 1)
        start_time = np.random.choice([start_time, start_time + 1], p=[0.2, 0.8])
        duration = np.random.randint(1, n_timesteps - start_time + 1)
        load_pattern = np.cumsum(np.random.uniform(0, 2, duration))
        delta_plus[i, start_time : start_time + duration] = load_pattern
    return delta_plus


def calculate_cost_per_kw2(delta_plus, change_cost_vec, remaining_load):
    """
    Calculate cost per kW² for each agent's contribution based on squared difference reduction.
    """
    n_submissions = delta_plus.shape[0]
    cost_per_kw2 = np.full(n_submissions, np.inf)

    # Calculate initial squared error
    initial_squared_error = np.sum(remaining_load**2)

    for i in range(n_submissions):
        # Calculate new squared error if we use this profile
        new_error = remaining_load - delta_plus[i]
        new_squared_error = np.sum(new_error**2)

        # Calculate improvement in squared error
        error_reduction = initial_squared_error - new_squared_error

        # Only consider profiles that reduce error and don't overshoot too much
        if error_reduction > 0 and np.all(new_error >= -0.1 * np.max(remaining_load)):
            cost_per_kw2[i] = change_cost_vec[i] / error_reduction

    return cost_per_kw2


def optimize_load_shape(
    load_target,
    Delta,
    costs,
    min_contribution=1e-10,
    bound_type="lower",
    plot: bool = True,
):
    """
    Optimize load shape using progressive load minimization with vectorized operations.

    Args:
        load_target (np.ndarray): Target load profile
        Delta (np.ndarray): 3D array of shape (n_timesteps, n_agents, n_timesteps) containing possible load modifications
        costs (np.ndarray): 2D array of shape (n_agents, n_timesteps) containing costs
        bound_type (str): Type of bound ('lower' or 'upper')
        plot (bool): Whether to plot the results

    Returns:
        tuple: (achieved_load, agents_dispatched, total_cost)
    """
    n_timesteps, n_agents, _ = Delta.shape
    achieved_load = np.zeros(n_timesteps)
    agents_dispatched = []
    offers_dispatched = []
    total_cost = 0

    # Create a mask for available agents
    available_agents = np.ones(n_agents, dtype=bool)

    while True:
        if not np.any(available_agents):
            break

        # Compute all remaining loads in one operation
        # Broadcasting to compute (achieved_load + Delta) for all timesteps and agents at once
        proposed_loads = achieved_load[None, None, :] + Delta
        remaining = load_target[None, None, :] - proposed_loads

        # Compute squared error for each possibility
        # Only consider available agents
        error_matrix = np.where(
            available_agents[None, :, None],
            np.where(remaining < 0, np.inf, remaining**2),
            np.inf,
        )

        # Sum errors across time dimension
        total_error = np.sum(error_matrix, axis=2)

        # Find the best combination
        if np.min(total_error) == np.inf:
            break

        best_timestep, best_agent = np.unravel_index(
            np.argmin(total_error), total_error.shape
        )

        # Verify if the chosen profile contributes anything
        if np.sum(np.abs(Delta[best_timestep, best_agent])) < min_contribution:
            available_agents[best_agent] = False
            continue

        # Update achieved load
        achieved_load += Delta[best_timestep, best_agent]

        # Mark agent as used
        available_agents[best_agent] = False
        agents_dispatched.append(best_agent)
        offers_dispatched.append(best_timestep)

        # Update total cost
        total_cost += costs[best_agent, best_timestep]

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(load_target, "b-", label="Target Load", linewidth=2)
        plt.plot(achieved_load, "r--", label="Achieved Load", linewidth=2)
        plt.fill_between(
            range(len(load_target)),
            load_target,
            achieved_load,
            color="gray",
            alpha=0.3,
            label="Error",
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Load")
        plt.legend()
        plt.grid(True)
        plt.title("Final Load Pattern Comparison")
        plt.show()

    return achieved_load, agents_dispatched, offers_dispatched, total_cost


def old_optimize_load_shape(
    load_target, Delta, costs, bound_type="lower", plot: bool = True
):
    """
    Optimize load shape using progressive cost per kW² minimization.
    """

    n_timesteps, n_agents, n_timesteps = Delta.shape
    achieved_load = np.zeros(n_timesteps)
    agents_already_selected = []
    total_cost = 0

    while True:
        # print(f"Target load: {load_target}")
        # print(f"Remaining load: {load_target - achieved_load}")
        # print(f"Achieved load: {achieved_load}")
        # print()
        Remaining = np.zeros(Delta.shape)
        for t in range(n_timesteps):
            for a in range(n_agents):
                if a not in agents_already_selected:
                    Remaining[t, a] = np.copy(load_target) - (
                        np.copy(Delta[t, a, :]) + np.copy(achieved_load)
                    )
                else:
                    Remaining[t, a] = np.inf

        # Convert remaining load to squared remaining load
        # First, set negative values to infinity
        Remaining[Remaining < 0] = np.inf
        Remaining_Squared = np.copy(Remaining) ** 2
        Remaining_Squared_Sum = np.sum(Remaining_Squared, axis=2).T
        # NOTE: Dim is (n_agents, n_timesteps)

        # Get the arguments indices of the global minimum
        if (
            np.min(Remaining_Squared_Sum) < np.inf
            and len(agents_already_selected) < n_agents
        ):
            min_args = np.unravel_index(
                Remaining_Squared_Sum.argmin(), Remaining_Squared_Sum.shape
            )

            best_agent = min_args[0]
            best_timestep_submission = min_args[1]

            # Verify if achieved load to be added is 0, break
            if np.sum(Delta[best_timestep_submission, best_agent, :]) == 0:
                break

            # Add chosen profile to the achieved load
            achieved_load += Delta[best_timestep_submission, best_agent, :]

            # Add agent to the list of selected agents
            agents_already_selected.append(best_agent)
        else:
            break

    if plot:
        # Plot final comparison
        plt.figure(figsize=(12, 6))
        plt.plot(load_target, "b-", label="Target Load", linewidth=2)
        plt.plot(achieved_load, "r--", label="Achieved Load", linewidth=2)
        plt.fill_between(
            range(len(load_target)),
            load_target,
            achieved_load,
            color="gray",
            alpha=0.3,
            label="Error",
        )
        plt.xlabel("Time Steps")
        plt.ylabel("Load")
        plt.legend()
        plt.grid(True)
        plt.title("Final Load Pattern Comparison")
        plt.show()

    return achieved_load, agents_already_selected, total_cost


# Example usage
if __name__ == "__main__":
    # Generate test data
    # n_submissions = 3000
    # n_timesteps = 24

    # Create a more challenging target load curve
    # x = np.linspace(0, np.pi / 1.5, n_timesteps)
    # load_target = 500 * np.sin(x)  # Sinusoidal target for better testing

    # delta_plus = generate_realistic_submissions(n_submissions, n_timesteps)
    # change_cost_vec = np.random.uniform(1, 10, n_submissions)

    n_timesteps = 12
    n_agents = 3000

    # load_target = np.array([10, 13, 14, 15, 15, 16])
    x = np.linspace(0, np.pi / 1.5, n_timesteps)
    load_target = 100 + 75 * np.sin(x)  # Sinusoidal target for better testing

    # Generate upper triangular matrices randomly
    agent_deltas = []
    for _ in range(n_agents):
        agent_deltas.append(
            np.triu(np.random.randint(0, 5, (n_timesteps, n_timesteps)))
        )

    # Define global Delta matrix (positive)
    Delta_plus = np.zeros((n_timesteps, n_agents, n_timesteps))
    for a in range(n_agents):
        for t in range(n_timesteps):
            Delta_plus[t, a, :] = np.copy(agent_deltas[a][t])

    costs = np.random.uniform(0.1, 10, (n_agents, n_timesteps))

    # Plot available load
    plot_available_load(Delta_plus, costs)
    # plot_executive_view(Delta_plus, costs)

    # Run optimization with visualization
    achieved_load, agents_dispatched, offers_dispatched, total_cost = (
        optimize_load_shape(
            load_target, Delta_plus, costs, bound_type="upper", plot=True
        )
    )
    print(f"Total cost: {total_cost}")
    print("AGENT DISPATCHED")
    print(agents_dispatched)
    print("AGENT OFFERS DISPATCHED")
    print(offers_dispatched)
