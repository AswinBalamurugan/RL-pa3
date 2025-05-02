from src.utils import *
from src.hyperparams import (
    MODELS_DIR,
    RESULTS_DIR,
    NUM_STATES,
    NUM_ACTIONS,
    GAMMA,
    ALPHA,
    EPSILON,
    NUM_RUNS,
    EPISODES,
    INTRAOPTIONS_MODEL_PREFIX,
    INTRAOPTIONS_REWARDS_FILE,
    INTRAOPTIONS_REWARDS_PLOT,
)

# Define color locations and navigation actions
color_targets = {
    "R": (0, 0),
    "G": (0, 4),
    "Y": (4, 0),
    "B": (4, 3),
}
navigation_actions = [0, 1, 2, 3]  # South, North, East, West
idx_to_color = ["R", "G", "Y", "B"]
total_actions = 10


def pretrain_navigation_options(env):
    print("\nPretraining navigation options...")
    option_policies = {}

    for color, target_loc in color_targets.items():
        Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        rewards = []

        for ep in range(5000):
            state, _ = env.reset()
            total_reward = 0

            for _ in range(200):
                taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
                if (taxi_row, taxi_col) == target_loc:
                    break

                if np.random.rand() < 0.001:
                    action = np.random.choice(navigation_actions)
                else:
                    action = navigation_actions[np.argmax(Q[state][navigation_actions])]

                next_state, _, terminated, truncated, _ = env.step(action)
                reward = -1

                # Update Q-table
                best_next = np.max(Q[next_state][navigation_actions])
                Q[state][action] += 0.2 * (reward + 0.9 * best_next - Q[state][action])
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        # Extract policy
        policy = np.zeros(NUM_STATES, dtype=int)
        for s in range(NUM_STATES):
            policy[s] = navigation_actions[np.argmax(Q[s][navigation_actions])]
        option_policies[color] = policy

    return option_policies


def option_terminates(state, color, env):
    taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
    return (taxi_row, taxi_col) == color_targets[color]


def is_consistent_option(state, action, color, option_policies):
    return option_policies[color][state] == action


def run_intraoptions():
    print("\n", "-" * 80)
    print("🚕 Starting Intra-Options Training Experiment...")
    logging.info("Starting Intra-Options experiment")
    env = gym.make("Taxi-v3")
    all_rewards = np.zeros((NUM_RUNS, EPISODES))

    # Pretrain navigation options
    option_policies = pretrain_navigation_options(env)

    best_avg = -np.inf
    best_model = None

    for run in range(NUM_RUNS):
        print(f"\n🏁 Starting Run {run + 1}/{NUM_RUNS}")
        logging.info(f"Starting run {run + 1}/{NUM_RUNS}")
        Q_main = np.zeros((NUM_STATES, total_actions))
        rewards_all = []

        with tqdm(range(1, EPISODES + 1), desc=f"Run {run+1}/{NUM_RUNS}") as pbar:
            for episode in pbar:
                epsilon = max(1e-3, EPSILON * (0.995**episode))
                state, _ = env.reset()
                done = False
                total_reward = 0

                while not done:
                    # Epsilon-greedy action selection
                    if np.random.rand() < epsilon:
                        # Get available actions (primitives + valid options)
                        available = list(range(6))
                        for i, color in enumerate(idx_to_color):
                            if not option_terminates(state, color, env):
                                available.append(i + 6)
                        action = np.random.choice(available)
                    else:
                        action = np.argmax(Q_main[state])

                    # Execute primitive action
                    if action < 6:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        # Update primitive action
                        Q_main[state][action] += ALPHA * (
                            reward
                            + GAMMA * np.max(Q_main[next_state])
                            - Q_main[state][action]
                        )

                        # Intra-option updates for consistent options
                        for i, color in enumerate(idx_to_color):
                            opt_idx = i + 6
                            if is_consistent_option(
                                state, action, color, option_policies
                            ):
                                if option_terminates(next_state, color, env):
                                    update = reward + GAMMA * np.max(Q_main[next_state])
                                else:
                                    update = (
                                        reward + GAMMA * Q_main[next_state][opt_idx]
                                    )
                                Q_main[state][opt_idx] += ALPHA * (
                                    update - Q_main[state][opt_idx]
                                )

                        total_reward += reward
                        state = next_state

                    # Execute option
                    else:
                        option_idx = action - 6
                        color = idx_to_color[option_idx]
                        start_state = state
                        option_reward = 0
                        steps = 0

                        while not done and steps < 50:
                            # Get primitive action from option policy
                            primitive_action = option_policies[color][state]
                            next_state, reward, terminated, truncated, _ = env.step(
                                primitive_action
                            )
                            done = terminated or truncated
                            option_reward += reward
                            steps += 1

                            # Update primitive action
                            Q_main[state][primitive_action] += ALPHA * (
                                reward
                                + GAMMA * np.max(Q_main[next_state])
                                - Q_main[state][primitive_action]
                            )

                            # Update current option
                            if option_terminates(next_state, color, env) or done:
                                update = reward + GAMMA * np.max(Q_main[next_state])
                            else:
                                update = reward + GAMMA * Q_main[next_state][action]
                            Q_main[start_state][action] += ALPHA * (
                                update - Q_main[start_state][action]
                            )

                            # Update other consistent options
                            for i, other_color in enumerate(idx_to_color):
                                other_opt_idx = i + 6
                                if is_consistent_option(
                                    state,
                                    primitive_action,
                                    other_color,
                                    option_policies,
                                ):
                                    if (
                                        option_terminates(next_state, other_color, env)
                                        or done
                                    ):
                                        update = reward + GAMMA * np.max(
                                            Q_main[next_state]
                                        )
                                    else:
                                        update = (
                                            reward
                                            + GAMMA * Q_main[next_state][other_opt_idx]
                                        )
                                    Q_main[state][other_opt_idx] += ALPHA * (
                                        update - Q_main[state][other_opt_idx]
                                    )

                            state = next_state
                            if option_terminates(state, color, env) or done:
                                break

                        total_reward += option_reward

                rewards_all.append(total_reward)
                all_rewards[run, episode - 1] = total_reward
                pbar.set_postfix({"Avg Reward (last 500)": np.mean(rewards_all[-500:])})

        # Calculate last 10% average rewards
        current_avg = np.mean(rewards_all[-EPISODES // 10 :])
        logging.info(
            f"Run {run + 1} completed. Average reward (last 10%): {current_avg}"
        )

        # Update best model if current run is better
        if current_avg > best_avg:
            best_avg = current_avg
            best_model = {
                "Q_main": Q_main,
                "option_policies": option_policies,
                "run": run,
                "rewards": rewards_all,
            }

    # Save best model and results
    if best_model is not None:
        model_path = os.path.join(MODELS_DIR, f"{INTRAOPTIONS_MODEL_PREFIX}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Best model saved to {model_path}")

    # Save final results
    rewards_path = os.path.join(RESULTS_DIR, "rewards", INTRAOPTIONS_REWARDS_FILE)
    plot_path = os.path.join(RESULTS_DIR, "plots", INTRAOPTIONS_REWARDS_PLOT)
    save_results(all_rewards, rewards_path)
    plot_rewards(
        all_rewards,
        plot_path,
        "Intra-option Q-Learning",
    )

    # Plot Q-values grid for the best model
    if best_model is not None:
        q_grid_path = os.path.join(
            RESULTS_DIR, "plots", "q_grid", "intraoptions_q_values_grid.png"
        )
        plot_q_values_and_policy(
            best_model["Q_main"],
            env=env,
            save_path=q_grid_path,
            action_names=[
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
            ],
        )
        logging.info(f"Q-values grid saved to {q_grid_path}")

    logging.info(f"Results saved to {plot_path} and {rewards_path}")
