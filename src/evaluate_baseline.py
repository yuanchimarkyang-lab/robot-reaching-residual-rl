"""
Experiments on Proportional Controller.

This script run evaluation experiments on proportional controller with Kp defined as in the configs/baseline.yaml file

"""

import gymnasium as gym
import gymnasium_robotics
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import goal_distance, save_video, summarize_episode
from config import load_config
from controllers import proportional_controller


def run_one_episode(env, kp, seed=None, record_video = False, success_threshold=0.05):
    """
    This function runs one experiment on a given kp, with an option to supply seed for reproduceability and record video. 
    The experiment results, including metrics, frames for video, actions, and achieved_goals, are returned.

    """
    obs, info = env.reset(seed=seed)

    frames = []
    distances = []
    achieved_goals = []
    rewards = []
    actions = []
    
    # initial distances after env.reset
    distances.append(goal_distance(obs))
    achieved_goals.append(obs["achieved_goal"])
    desired_goal = obs["desired_goal"]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if record_video:
            frames.append(env.render())

        action = proportional_controller(obs, kp=kp)

        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(goal_distance(obs))
        achieved_goals.append(obs["achieved_goal"])
        rewards.append(reward)
        actions.append(action)

    metrics = summarize_episode(distances=distances, 
                                rewards=rewards, 
                                actions=actions, 
                                success_threshold=success_threshold)
    metrics["initial_distance"] = distances[0]
    metrics["initial_achieved"] = achieved_goals[0]
    metrics["initial_desired"] = desired_goal
    return metrics, frames, actions, achieved_goals





def main():
    gym.register_envs(gymnasium_robotics)

    env_config = load_config("configs/env.yaml")
    baseline_config = load_config("configs/baseline.yaml")

    env = gym.make(
        env_config["env_id"],
        render_mode=env_config["render_mode"],
    )
    
    kp_values = baseline_config["kp_values"]
    num_episodes = baseline_config["num_eval_episodes"]
    success_threshold = baseline_config["success_threshold"]
    base_seed = env_config["seed"]
    
    output_path = Path("results/metrics/baseline/proportional_controller_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for kp in kp_values:
        print(f"\nEvaluating proportional controller with kp={kp}")
        
        for episode_idx in tqdm(range(num_episodes)):
            record_video = (baseline_config["record_video"] and (episode_idx in {0,1,14}))

            metrics, frames, actions, achieved_goals = run_one_episode(
                env=env,
                kp=kp,
                seed=base_seed + episode_idx, # the same seeds are used to enhance reproducebility
                record_video=record_video,
                success_threshold=success_threshold
            )

            metrics["episode"] = episode_idx
            metrics["policy"] = "proportional"
            metrics["kp"] = kp
            metrics["env_id"] = env_config["env_id"]

            all_metrics.append(metrics)
            
            episode_label = "000"+str(episode_idx)
            episode_label = episode_label[-3:]
            if record_video:
                save_video(
                    frames,
                    f"results/videos/proportional_kp_{kp}_episode_{episode_label}.mp4",
                    fps=3,
                )
            
            # actions and achieved goals are stored for error analysis
            actions = np.array(actions)
            np.save(f"results/metrics/baseline/actions_proportional_kp_{kp}_episode_{episode_label}.npy", actions)
            achieved_goals = np.array(achieved_goals)
            np.save(f"results/metrics/baseline/achieved_goals_proportional_kp_{kp}_episode_{episode_label}.npy", achieved_goals)
    env.close()

    df = pd.DataFrame(all_metrics)

    df.to_csv(output_path, index=False)

    print("\nProportional controller evaluation complete.")
    print(f"\nSuccessful Threshold: {success_threshold} (m)")
    print("\nSummary by kp")
    summary = df.groupby("kp").agg(
        success_rate=("success", "mean"),
        mean_final_distance=("final_distance","mean"),
        median_final_distance=("final_distance","median"),
        mean_return=("total_return", "mean"),
        mean_action_norm=("mean_action_norm","mean"),
        mean_action_smoothness=("action_smoothness","mean")
    )

    print(summary)
    print(f"\nSaved metrics to: {output_path}")

if __name__ == "__main__":
    main()

