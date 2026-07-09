"""
The experiment of random rollout.

This script runs the random rollout as sanity check. The random rollout is deinfed as actions sampled randomly at the action_space at the given environment.

"""
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import goal_distance, save_video, summarize_episode
from config import load_config


def run_one_episode(env, seed=None, record_video = False):
    """
    This function runs one experiment on random rollout, with an option to supply seed for reproduceability and record video. 
    The experiment results, including metrics and frames for video are returned.

    """
    obs, info = env.reset(seed=seed)

    frames = []
    distances = []
    rewards = []
    actions = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if record_video:
            frames.append(env.render())

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(goal_distance(obs))
        rewards.append(reward)
        actions.append(action)

    metrics = summarize_episode(distances, rewards, actions)

    return metrics, frames





def main():
    gym.register_envs(gymnasium_robotics)

    config = load_config()

    env = gym.make(
        config["env_id"],
        render_mode=config["render_mode"],
    )

    num_episodes = config["num_eval_episodes"]
    base_seed = config["seed"]

    all_metrics = []

    for episode_idx in tqdm(range(num_episodes)):
        record_video = episode_idx == 0

        metrics, frames = run_one_episode(
            env,
            seed=base_seed + episode_idx,
            record_video=record_video,
        )

        metrics["episode"] = episode_idx
        metrics["policy"] = "random"
        metrics["env_id"] = config["env_id"]

        all_metrics.append(metrics)

        if record_video:
            save_video(
                frames,
                "results/videos/random_policy_episode_000.mp4",
                fps=30,
            )

    env.close()

    df = pd.DataFrame(all_metrics)

    output_path = Path("results/metrics/random_policy_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nRandom policy evaluation complete.")
    #print(f"\nSuccess Threshold: {success_threshold} (m)")
    print(df.describe())
    print(f"\nSaved metrics to: {output_path}")

if __name__ == "__main__":
    main()

