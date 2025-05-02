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
    SMDP_MODEL_PREFIX,
    SMDP_REWARDS_FILE,
    SMDP_REWARDS_PLOT,
)
import logging


def run_smdp():
    print("\n", "-" * 80)
    print("🚕 Starting SMDP Training Experiment...")
    logging.info("Starting SMDP experiment")
    env = gym.make("Taxi-v3")
    all_rewards = np.zeros((NUM_RUNS, EPISODES))

    best_avg = -np.inf
    best_model = None

    # Define color locations
    color_targets = {
        "R": (0, 0),
        "G": (0, 4),
        "Y": (4, 0),
        "B": (4, 3),
    }

    # Pretrain navigation options
    print("Pretraining navigation options...")
    option_Q_tables = {}
    for color, target_loc in color_targets.items():
        Q = np.zeros(
            (NUM_STATES, NUM_ACTIONS)
        )  # Full action space, but we use only 0-3
        rewards = []

        for ep in range(EPISODES // 10):  # Pretrain for 10% of total episodes
            state, _ = env.reset()
            total_reward = 0

            for step in range(100):
                taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)

                if (taxi_row, taxi_col) == target_loc:
                    reward = 10
                    total_reward += reward
                    break  # End option
                else:
                    if np.random.rand() < EPSILON:
                        action = np.random.randint(0, 4)  # Only navigation actions
                    else:
                        action = np.argmax(Q[state][0:4])  # Only navigation actions

                    next_state, _, terminated, truncated, _ = env.step(action)
                    reward = -1

                    best_next = np.argmax(Q[next_state][0:4])
                    Q[state][action] += ALPHA * (
                        reward + GAMMA * Q[next_state][best_next] - Q[state][action]
                    )
                    total_reward += reward
                    state = next_state

                    if terminated or truncated:
                        break

            rewards.append(total_reward)

        option_Q_tables[color] = Q

    # Main training loop
    for run in range(NUM_RUNS):
        print(f"\n🏁 Starting Run {run + 1}/{NUM_RUNS}")
        logging.info(f"Starting run {run + 1}/{NUM_RUNS}")
        Q_main = np.zeros((NUM_STATES, 10))  # 6 primitive actions + 4 options
        rewards_all = []

        with tqdm(range(1, EPISODES + 1), desc=f"Run {run+1}/{NUM_RUNS}") as pbar:
            for episode in pbar:
                epsilon = max(1e-3, EPSILON * (0.995**episode))
                state, _ = env.reset()
                done = False
                total_reward = 0

                while not done:
                    # Choose action or option
                    if random.random() < epsilon:
                        action = random.randint(0, 9)  # 6 primitive + 4 options
                    else:
                        action = np.argmax(Q_main[state])

                    if action < 6:
                        # Primitive action
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        # Standard Q-learning update
                        Q_main[state][action] += ALPHA * (
                            reward
                            + GAMMA * np.max(Q_main[next_state])
                            - Q_main[state][action]
                        )

                        state = next_state
                        total_reward += reward

                    else:
                        # Option execution (navigation only)
                        option_idx = action - 6
                        option_color = list(color_targets.keys())[option_idx]
                        target_row, target_col = color_targets[option_color]
                        option_Q = option_Q_tables[option_color]

                        curr_state = state
                        duration = 0
                        discounted_rewards = (
                            0  # To accumulate properly discounted rewards
                        )

                        while True:
                            action_values = option_Q[curr_state][
                                0:4
                            ]  # Only movement actions
                            primitive_action = np.argmax(action_values)
                            next_state, reward, terminated, truncated, _ = env.step(
                                primitive_action
                            )

                            taxi_row, taxi_col, _, _ = env.unwrapped.decode(next_state)
                            at_target = (taxi_row == target_row) and (
                                taxi_col == target_col
                            )

                            # Properly discount the reward at each step
                            discounted_rewards += (GAMMA**duration) * reward
                            duration += 1
                            curr_state = next_state
                            done = terminated or truncated

                            if done or at_target or duration > 50:
                                break

                        # SMDP Q-update for option
                        Q_main[state][action] += ALPHA * (
                            discounted_rewards
                            + (GAMMA**duration) * np.max(Q_main[curr_state])
                            - Q_main[state][action]
                        )

                        state = curr_state
                        total_reward += (
                            discounted_rewards  # Add discounted rewards to total
                        )

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
                "option_Q_tables": option_Q_tables,
                "run": run,
                "rewards": rewards_all,
            }

    # Save only the best model
    if best_model is not None:
        model_path = os.path.join(MODELS_DIR, f"{SMDP_MODEL_PREFIX}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Best model saved to {model_path}")

    # Save final results
    rewards_path = os.path.join(RESULTS_DIR, "rewards", SMDP_REWARDS_FILE)
    plot_path = os.path.join(RESULTS_DIR, "plots", SMDP_REWARDS_PLOT)
    save_results(all_rewards, rewards_path)
    plot_rewards(all_rewards, plot_path, "SMDP Q-Learning")

    # Plot Q-values grid for the best model
    if best_model is not None:
        q_grid_path = os.path.join(
            RESULTS_DIR, "plots", "q_grid", "smdp_q_values_grid.png"
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

    logging.info(f"Results saved to {rewards_path} and {plot_path}")
