import gymnasium as gym
import numpy as np
import time
from IPython.display import clear_output
from src.utils import *
from src.hyperparams import (
    SMDP_MODEL_PREFIX,
    INTRAOPTIONS_MODEL_PREFIX,
    MODELS_DIR,
)
import logging


def render_model(model_type):
    # Determine the model path based on the model type
    if model_type == "smdp":
        model_path = os.path.join(MODELS_DIR, f"{SMDP_MODEL_PREFIX}_best.pkl")
    elif model_type == "intraoptions":
        model_path = os.path.join(MODELS_DIR, f"{INTRAOPTIONS_MODEL_PREFIX}_best.pkl")
    else:
        raise ValueError("Invalid model type. Choose 'smdp' or 'intraoptions'.")

    # Load the model
    model = load_model(model_path)
    Q_main = model["Q_main"]
    Q_options = model["Q_options"]

    rewards = []

    logging.info(f"Starting Taxi Evaluation for {model_type.upper()} Model...")
    print(f"\n🚕 Starting Taxi Evaluation for {model_type.upper()} Model...\n")

    for run in range(5):
        # Initialize the environment
        env = gym.make("Taxi-v3", render_mode="human")
        state, _ = env.reset()
        done = False
        total_reward = 0

        logging.info(f"Starting evaluation run {run + 1}")
        print(f"🏁 Starting evaluation run {run + 1}...")

        while not done:
            # Select best option for current state
            option = np.argmax(Q_main[state])
            logging.debug(f"Selected Option {option} for state {state}")

            curr_state = state
            # Execute the chosen option until it terminates
            while True:
                action = np.argmax(Q_options[option][curr_state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                curr_state = next_state
                done = terminated or truncated

                time.sleep(0.4)

                # If the option's primitive (pickup=4 or dropoff=5) fires or env ends, stop
                if done or action in [4, 5]:
                    logging.debug(f"Option terminated due to action {action}")
                    break

            state = curr_state

        rewards.append(total_reward)
        logging.info(f"Run {run + 1} completed. Total reward: {total_reward}")
        print(f"🏁 Finished evaluation run {run+1}. Total reward: {total_reward}")
        env.close()
        time.sleep(1)

    logging.info("All evaluation runs completed.")
    print("-" * 80)
    print("Final rewards for all runs:")
    for i, reward in enumerate(rewards):
        logging.info(f"Run {i+1}: {reward}")
        print(f"Run {i+1}: {reward}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        filename="logs/render.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Select model to render:")
    print("1. SMDP")
    print("2. Intra-Options")

    while True:
        try:
            choice = int(input("Enter your choice (1 or 2): "))
            if choice == 1:
                render_model("smdp")
                break
            elif choice == 2:
                render_model("intraoptions")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
