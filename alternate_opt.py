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
)

# Define file names for the alternate options
ALT_SMDP_MODEL_PREFIX = "alt_smdp"
ALT_SMDP_REWARDS_FILE = "alt_smdp_rewards.csv"
ALT_SMDP_REWARDS_PLOT = "alt_smdp_rewards.png"
ALT_INTRAOPTIONS_MODEL_PREFIX = "alt_intraoptions"
ALT_INTRAOPTIONS_REWARDS_FILE = "alt_intraoptions_rewards.csv"
ALT_INTRAOPTIONS_REWARDS_PLOT = "alt_intraoptions_rewards.png"
COMPARISON_PLOT = "options_comparison.png"


# Function to run the SMDP Q-learning with alternate options
def run_alt_smdp():
    print("\n", "-" * 80)
    print("🚕 Starting SMDP Training with Alternate Options...")
    logging.info("Starting SMDP experiment with alternate options")

    env = gym.make("Taxi-v3")
    all_rewards = np.zeros((NUM_RUNS, EPISODES))
    best_avg = -np.inf
    best_model = None

    # Define boundary targets
    boundary_targets = {
        "TOP_ROW": 0,  # Row 0 (top)
        "BOTTOM_ROW": 4,  # Row 4 (bottom)
        "LEFT_COL": 0,  # Column 0 (leftmost)
        "RIGHT_COL": 4,  # Column 4 (rightmost)
    }

    # Pretrain boundary navigation options
    print("Pretraining boundary navigation options...")
    option_Q_tables = {}
    for boundary_name, target_idx in boundary_targets.items():
        Q = np.zeros(
            (NUM_STATES, NUM_ACTIONS)
        )  # Full action space, but we use only 0-3
        rewards = []

        for ep in range(EPISODES // 10):  # Pretrain for 10% of total episodes
            state, _ = env.reset()
            total_reward = 0

            for step in range(100):
                taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)

                # Check if we've reached the boundary
                at_boundary = False
                if boundary_name == "TOP_ROW" and taxi_row == target_idx:
                    at_boundary = True
                elif boundary_name == "BOTTOM_ROW" and taxi_row == target_idx:
                    at_boundary = True
                elif boundary_name == "LEFT_COL" and taxi_col == target_idx:
                    at_boundary = True
                elif boundary_name == "RIGHT_COL" and taxi_col == target_idx:
                    at_boundary = True

                if at_boundary:
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

        option_Q_tables[boundary_name] = Q

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
                        # Option execution (boundary navigation)
                        option_idx = action - 6
                        boundary_name = list(boundary_targets.keys())[option_idx]
                        boundary_target = boundary_targets[boundary_name]
                        option_Q = option_Q_tables[boundary_name]

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

                            # Check if we've reached the boundary
                            at_boundary = False
                            if (
                                boundary_name == "TOP_ROW"
                                and taxi_row == boundary_target
                            ):
                                at_boundary = True
                            elif (
                                boundary_name == "BOTTOM_ROW"
                                and taxi_row == boundary_target
                            ):
                                at_boundary = True
                            elif (
                                boundary_name == "LEFT_COL"
                                and taxi_col == boundary_target
                            ):
                                at_boundary = True
                            elif (
                                boundary_name == "RIGHT_COL"
                                and taxi_col == boundary_target
                            ):
                                at_boundary = True

                            # Properly discount the reward at each step
                            discounted_rewards += (GAMMA**duration) * reward
                            duration += 1
                            curr_state = next_state
                            done = terminated or truncated

                            if done or at_boundary or duration > 50:
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
        print(f"📈 Average reward (last 10%): {current_avg:.2f}\n")
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
        model_path = os.path.join(MODELS_DIR, f"{ALT_SMDP_MODEL_PREFIX}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Best model saved to {model_path}")

        # Plot Q-values grid for the best model
        q_grid_path = os.path.join(
            RESULTS_DIR, "plots", "q_grid", "alt_smdp_q_values_grid.png"
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
                "go to TOP",
                "go to BOTTOM",
                "go to LEFT",
                "go to RIGHT",
            ],
        )
        logging.info(f"Q-values grid saved to {q_grid_path}")

    # Save final results
    rewards_path = os.path.join(RESULTS_DIR, "rewards", ALT_SMDP_REWARDS_FILE)
    plot_path = os.path.join(RESULTS_DIR, "plots", ALT_SMDP_REWARDS_PLOT)
    save_results(all_rewards, rewards_path)
    plot_rewards(all_rewards, plot_path, "SMDP Q-Learning with Boundary Options")
    logging.info(f"Results saved to {rewards_path} and {plot_path}")

    return all_rewards


def run_alt_intraoptions():
    print("\n", "-" * 80)
    print("🚕 Starting Intra-Options Training with Alternate Options...")
    logging.info("Starting Intra-Options experiment with alternate options")

    env = gym.make("Taxi-v3")
    all_rewards = np.zeros((NUM_RUNS, EPISODES))
    best_avg = -np.inf
    best_model = None

    # Define boundary targets
    boundary_targets = {
        "TOP_ROW": 0,  # Row 0 (top)
        "BOTTOM_ROW": 4,  # Row 4 (bottom)
        "LEFT_COL": 0,  # Column 0 (leftmost)
        "RIGHT_COL": 4,  # Column 4 (rightmost)
    }

    # Pretrain boundary navigation options
    print("Pretraining boundary navigation options...")
    option_Q_tables = {}
    option_policies = {}
    for boundary_name, target_idx in boundary_targets.items():
        Q = np.zeros(
            (NUM_STATES, NUM_ACTIONS)
        )  # Full action space, but we use only 0-3
        rewards = []

        for ep in range(EPISODES // 10):  # Pretrain for 10% of total episodes
            state, _ = env.reset()
            total_reward = 0

            for step in range(100):
                taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)

                # Check if we've reached the boundary
                at_boundary = False
                if boundary_name == "TOP_ROW" and taxi_row == target_idx:
                    at_boundary = True
                elif boundary_name == "BOTTOM_ROW" and taxi_row == target_idx:
                    at_boundary = True
                elif boundary_name == "LEFT_COL" and taxi_col == target_idx:
                    at_boundary = True
                elif boundary_name == "RIGHT_COL" and taxi_col == target_idx:
                    at_boundary = True

                if at_boundary:
                    reward = 0  # No extra reward for reaching target
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

        option_Q_tables[boundary_name] = Q

        # Extract policy
        policy = np.zeros(NUM_STATES, dtype=int)
        for s in range(NUM_STATES):
            policy[s] = np.argmax(Q[s][0:4])  # Only navigation actions
        option_policies[boundary_name] = policy

    # Function to check if option should terminate
    def option_terminates(state, boundary_name):
        taxi_row, taxi_col, _, _ = env.unwrapped.decode(state)
        target_idx = boundary_targets[boundary_name]

        if boundary_name == "TOP_ROW":
            return taxi_row == target_idx
        elif boundary_name == "BOTTOM_ROW":
            return taxi_row == target_idx
        elif boundary_name == "LEFT_COL":
            return taxi_col == target_idx
        elif boundary_name == "RIGHT_COL":
            return taxi_col == target_idx
        return False

    # Function to get the action prescribed by an option
    def get_option_action(state, boundary_name):
        return option_policies[boundary_name][state]

    # Function to check if an option would take the same action
    def is_consistent_option(state, action, boundary_name):
        return option_policies[boundary_name][state] == action

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
                    # Choose action or option with epsilon-greedy
                    if np.random.rand() < epsilon:
                        # Get available actions (primitives + valid options)
                        available = list(range(6))
                        for i, boundary_name in enumerate(boundary_targets.keys()):
                            if not option_terminates(state, boundary_name):
                                available.append(i + 6)
                        action = np.random.choice(available)
                    else:
                        action = np.argmax(Q_main[state])

                    # Execute primitive action
                    if action < 6:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        # Standard Q-learning update for the chosen primitive action
                        Q_main[state][action] += ALPHA * (
                            reward
                            + GAMMA * np.max(Q_main[next_state])
                            - Q_main[state][action]
                        )

                        # Intra-option updates for consistent options
                        for i, boundary_name in enumerate(boundary_targets.keys()):
                            opt_idx = i + 6
                            if is_consistent_option(state, action, boundary_name):
                                if option_terminates(next_state, boundary_name):
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
                        boundary_name = list(boundary_targets.keys())[option_idx]
                        start_state = state
                        option_reward = 0
                        steps = 0

                        while not done and steps < 50:
                            # Get primitive action from option policy
                            primitive_action = get_option_action(state, boundary_name)
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
                            if option_terminates(next_state, boundary_name) or done:
                                update = reward + GAMMA * np.max(Q_main[next_state])
                            else:
                                update = reward + GAMMA * Q_main[next_state][action]
                            Q_main[start_state][action] += ALPHA * (
                                update - Q_main[start_state][action]
                            )

                            # Update other consistent options
                            for j, other_boundary in enumerate(boundary_targets.keys()):
                                other_opt_idx = j + 6
                                if is_consistent_option(
                                    state, primitive_action, other_boundary
                                ):
                                    if (
                                        option_terminates(next_state, other_boundary)
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
                            if option_terminates(state, boundary_name) or done:
                                break

                        total_reward += option_reward

                rewards_all.append(total_reward)
                all_rewards[run, episode - 1] = total_reward
                pbar.set_postfix({"Avg Reward (last 500)": np.mean(rewards_all[-500:])})

        # Calculate last 10% average rewards
        current_avg = np.mean(rewards_all[-EPISODES // 10 :])
        print(f"📈 Average reward (last 10%): {current_avg:.2f}\n")
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
        model_path = os.path.join(
            MODELS_DIR, f"{ALT_INTRAOPTIONS_MODEL_PREFIX}_best.pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Best model saved to {model_path}")

        # Plot Q-values grid for the best model
        q_grid_path = os.path.join(
            RESULTS_DIR, "plots", "q_grid", "alt_intraoptions_q_values_grid.png"
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
                "go to TOP",
                "go to BOTTOM",
                "go to LEFT",
                "go to RIGHT",
            ],
        )
        logging.info(f"Q-values grid saved to {q_grid_path}")

    # Save final results
    rewards_path = os.path.join(RESULTS_DIR, "rewards", ALT_INTRAOPTIONS_REWARDS_FILE)
    plot_path = os.path.join(RESULTS_DIR, "plots", ALT_INTRAOPTIONS_REWARDS_PLOT)
    save_results(all_rewards, rewards_path)
    plot_rewards(
        all_rewards,
        plot_path,
        "Intra-option Q-Learning with Alternate Options",
    )
    logging.info(f"Results saved to {rewards_path} and {plot_path}")

    return all_rewards


# Function to compare original and alternate options
def compare_options():
    print("\n", "-" * 80)
    print("📊 Comparing Original and Alternate Options...")

    # Load original results
    original_smdp_file = os.path.join(RESULTS_DIR, "rewards", "smdp_rewards.csv")
    original_intraoptions_file = os.path.join(
        RESULTS_DIR, "rewards", "intraoptions_rewards.csv"
    )

    # Load alternate results
    alternate_smdp_file = os.path.join(RESULTS_DIR, "rewards", ALT_SMDP_REWARDS_FILE)
    alternate_intraoptions_file = os.path.join(
        RESULTS_DIR, "rewards", ALT_INTRAOPTIONS_REWARDS_FILE
    )

    # Check if all files exist
    files_exist = all(
        os.path.exists(f)
        for f in [
            original_smdp_file,
            original_intraoptions_file,
            alternate_smdp_file,
            alternate_intraoptions_file,
        ]
    )

    if not files_exist:
        print(
            "⚠️ Not all results files exist. Run both original and alternate options first."
        )
        return

    # Load the results using pandas
    original_smdp_rewards = pd.read_csv(original_smdp_file).values
    original_intraoptions_rewards = pd.read_csv(original_intraoptions_file).values
    alternate_smdp_rewards = pd.read_csv(alternate_smdp_file).values
    alternate_intraoptions_rewards = pd.read_csv(alternate_intraoptions_file).values

    # Calculate the average rewards across runs
    orig_smdp_avg = original_smdp_rewards.mean(axis=1)
    orig_intra_avg = original_intraoptions_rewards.mean(axis=1)
    alt_smdp_avg = alternate_smdp_rewards.mean(axis=1)
    alt_intra_avg = alternate_intraoptions_rewards.mean(axis=1)

    # Create a comparison plot
    plt.figure(figsize=(12, 8))

    # Apply smoothing with a moving average
    window_size = min(100, EPISODES // 10)
    print(f"Using smoothing window size: {window_size}")
    smoothing_window = np.ones(window_size) / window_size

    orig_smdp_smooth = np.convolve(orig_smdp_avg, smoothing_window, mode="valid")
    orig_intra_smooth = np.convolve(orig_intra_avg, smoothing_window, mode="valid")
    alt_smdp_smooth = np.convolve(alt_smdp_avg, smoothing_window, mode="valid")
    alt_intra_smooth = np.convolve(alt_intra_avg, smoothing_window, mode="valid")

    # Create x-axis values (episodes) accounting for smoothing
    episodes = np.arange(window_size - 1, EPISODES)

    # Plot the smoothed rewards
    plt.plot(episodes, orig_smdp_smooth, label="Original SMDP", color="blue")
    plt.plot(episodes, orig_intra_smooth, label="Original Intra-Options", color="green")
    plt.plot(episodes, alt_smdp_smooth, label="Alternate SMDP", color="red")
    plt.plot(
        episodes, alt_intra_smooth, label="Alternate Intra-Options", color="purple"
    )

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Comparison of Original and Alternate Options")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the comparison plot
    comparison_path = os.path.join(RESULTS_DIR, "plots", COMPARISON_PLOT)
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()

    print(f"📈 Comparison plot saved to {comparison_path}")

    # Print summary statistics
    print("\n📊 Summary Statistics (Last 10% of Episodes):")
    last_10_percent = EPISODES // 10

    orig_smdp_final = np.mean(orig_smdp_avg[-last_10_percent:])
    orig_intra_final = np.mean(orig_intra_avg[-last_10_percent:])
    alt_smdp_final = np.mean(alt_smdp_avg[-last_10_percent:])
    alt_intra_final = np.mean(alt_intra_avg[-last_10_percent:])

    print(f"Original SMDP: {orig_smdp_final:.2f}")
    print(f"Original Intra-Options: {orig_intra_final:.2f}")
    print(f"Alternate SMDP: {alt_smdp_final:.2f}")
    print(f"Alternate Intra-Options: {alt_intra_final:.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/alternate_options.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("🚀 Alternate Options Experiment")
    print("1. Run SMDP with Alternate Options")
    print("2. Run Intra-Options with Alternate Options")
    print("3. Run Both and Compare with Alternate Options")
    print("4. Compare Results (if all experiments have been run)")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        run_alt_smdp()
    elif choice == "2":
        run_alt_intraoptions()
    elif choice == "3":
        print("Running SMDP with Alternate Options...")
        smdp_rewards = run_alt_smdp()

        print("Running Intra-Options with Alternate Options...")
        intraoptions_rewards = run_alt_intraoptions()

        print("Comparing all results...")
        compare_options()
    elif choice == "4":
        compare_options()
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
