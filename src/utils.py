import numpy as np
import logging
import os
import gymnasium as gym
import pandas as pd
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scienceplots

warnings.filterwarnings("ignore")

plt.style.use(["ieee", "no-latex"])
sns.set_palette("colorblind")
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.titlesize"] = 20
sns.set_style("dark")


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_results(rewards, file_path):
    columns = [f"Run_{i+1}" for i in range(rewards.shape[0])]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame(rewards.T, columns=columns)
    df.to_csv(file_path, index=False)
    logging.info(f"Saved results to {file_path}")


def plot_rewards(rewards, file_path, model):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    WINDOW_SIZE = 50

    for i, reward in enumerate(rewards):
        ma = np.convolve(reward, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode="valid")
        plt.plot(ma, alpha=0.5, label=f"Run {i+1} Moving Avg")

    mean_rewards = np.mean(rewards, axis=0)
    mean_ma = np.convolve(
        mean_rewards, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode="valid"
    )
    plt.plot(mean_ma, color="black", linewidth=2, label="Mean Moving Avg")

    plt.title(f"{model}: Moving Average Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def decode_taxi_state(state):
    # Taxi-v3: 500 states = 5 rows * 5 cols * 5 passenger locations * 4 destinations
    out = []
    out.append(state % 5)  # taxi_col
    state //= 5
    out.append(state % 5)  # taxi_row
    state //= 5
    out.append(state % 5)  # passenger_location (0-3: R,G,Y,B, 4: in taxi)
    state //= 5
    out.append(state % 4)  # destination (0-3: R,G,Y,B)
    # Return in (row, col, passenger, dest) order for clarity
    return (out[1], out[0], out[2], out[3])


def plot_q_values_and_policy(Q, env, save_path=None, action_names=None):
    """
    Plots Q-values and learned policies for Taxi-v3 environment in a grid format.
    Creates visualizations showing both Q-values and corresponding policies for
    different passenger locations and destinations.

    Args:
        Q: Q-table with shape (num_states, num_actions)
        env: Taxi environment instance
        save_path: Path to save the figure
        action_names: List of action names (default: South, North, East, West, Pickup, Dropoff, R, G, Y, B)
    """

    if action_names is None:
        action_names = [
            "south",
            "north",
            "east",
            "west",
            "pick\npassenger",
            "drop\npassenger",
            "go to R",
            "go to G",
            "go to Y",
            "go to B",
        ]

    # Define destinations and passenger locations to visualize
    destinations = ["R", "G", "Y", "B"]
    dest_indices = [0, 1, 2, 3]  # R, G, Y, B indices
    pass_locations = ["R", "In Taxi"]
    pass_indices = [0, 4]  # R index and In Taxi index

    # Create a 4x4 grid of subplots (4 destinations x 4 visualizations)
    _, axes = plt.subplots(4, 4, figsize=(20, 20))

    # Create colormaps
    q_cmap = sns.color_palette("vlag", as_cmap=True)
    policy_cmap = sns.color_palette("coolwarm", as_cmap=True)

    for row, (dest, dest_idx) in enumerate(zip(destinations, dest_indices)):
        # First column: Q-values for passenger at R
        ax = axes[row, 0]
        q_grid = np.zeros((5, 5))

        # Fill Q-value grid for passenger at R, going to destination
        for taxi_row in range(5):
            for taxi_col in range(5):
                state_idx = env.unwrapped.encode(
                    taxi_row, taxi_col, pass_indices[0], dest_idx
                )
                q_grid[taxi_row, taxi_col] = np.max(Q[state_idx])

        # Plot Q-values
        sns.heatmap(q_grid, ax=ax, cmap=q_cmap, annot=True, fmt=".1f", linewidths=0.5)
        ax.set_title(
            f"passenger location: R, destination:{dest}\nQ values for state",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Second column: Policy for passenger at R
        ax = axes[row, 1]
        policy_grid = np.zeros((5, 5))
        policy_actions = np.empty((5, 5), dtype=object)

        for taxi_row in range(5):
            for taxi_col in range(5):
                state_idx = env.unwrapped.encode(
                    taxi_row, taxi_col, pass_indices[0], dest_idx
                )
                best_action = np.argmax(Q[state_idx])
                policy_grid[taxi_row, taxi_col] = np.max(Q[state_idx])

                if best_action < len(action_names):
                    policy_actions[taxi_row, taxi_col] = action_names[best_action]
                else:
                    policy_actions[taxi_row, taxi_col] = "unknown"

        # Plot policy
        sns.heatmap(
            policy_grid,
            ax=ax,
            cmap=policy_cmap,
            annot=policy_actions,
            fmt="",
            linewidths=0.5,
            annot_kws={"size": 8},
        )
        ax.set_title(
            f"passenger location: R, destination:{dest}\nlearned policy", fontsize=10
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Third column: Q-values for passenger in taxi
        ax = axes[row, 2]
        q_grid = np.zeros((5, 5))

        # Fill Q-value grid for passenger in taxi
        for taxi_row in range(5):
            for taxi_col in range(5):
                state_idx = env.unwrapped.encode(
                    taxi_row, taxi_col, pass_indices[1], dest_idx
                )
                q_grid[taxi_row, taxi_col] = np.max(Q[state_idx])

        # Plot Q-values
        sns.heatmap(q_grid, ax=ax, cmap=q_cmap, annot=True, fmt=".1f", linewidths=0.5)
        ax.set_title(
            f"passenger location: In Taxi, destination:{dest}\nQ values for state",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Fourth column: Policy for passenger in taxi
        ax = axes[row, 3]
        policy_grid = np.zeros((5, 5))
        policy_actions = np.empty((5, 5), dtype=object)

        for taxi_row in range(5):
            for taxi_col in range(5):
                state_idx = env.unwrapped.encode(
                    taxi_row, taxi_col, pass_indices[1], dest_idx
                )
                best_action = np.argmax(Q[state_idx])
                policy_grid[taxi_row, taxi_col] = np.max(Q[state_idx])

                if best_action < len(action_names):
                    policy_actions[taxi_row, taxi_col] = action_names[best_action]
                else:
                    policy_actions[taxi_row, taxi_col] = "unknown"

        # Plot policy
        sns.heatmap(
            policy_grid,
            ax=ax,
            cmap=policy_cmap,
            annot=policy_actions,
            fmt="",
            linewidths=0.5,
            annot_kws={"size": 8},
        )
        ax.set_title(
            f"passenger location: In Taxi, destination:{dest}\nlearned policy",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
