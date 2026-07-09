"""
Experiments on residual SAC.

This script evaluate the performance of trained residual SAC model with evaluation setup and mode path defined in the configs/residual_sac.yaml file.

"""
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from tqdm import tqdm

from config import load_config
from env_utils import make_fetch_env
from utils import goal_distance, save_video, summarize_episode
from wrappers import ResidualActionWrapper



def run_one_episode(
    env,
    model,
    seed=None,
    record_video=False,
    success_threshold=0.05,
):
    """
    This function runs one experiment on a given model, with an option to supply seed for reproduceability and record video. 
    The experiment results, including metrics, frames for video, residual_actions, baseline_actions, final_actions and achieved_goals, are returned.

    """
    obs, info = env.reset(seed=seed)

    frames = []
    distances = []
    rewards = []

    residual_actions = []
    baseline_actions = []
    final_actions = []
    achieved_goals = []

    distances.append(goal_distance(obs))
    achieved_goals.append(obs["achieved_goal"])
    desired_goal = obs["desired_goal"]
    
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if record_video:
            frames.append(env.render())

        residual_action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(residual_action)

        distances.append(goal_distance(obs))
        achieved_goals.append(obs["achieved_goal"])
        rewards.append(reward)

        residual_actions.append(env.last_residual_action)
        baseline_actions.append(env.last_baseline_action)
        final_actions.append(env.last_final_action)


    metrics = summarize_episode(
        distances=distances,
        rewards=rewards,
        actions=final_actions,
        success_threshold=success_threshold,
    )

    residual_actions = np.array(residual_actions)
    baseline_actions = np.array(baseline_actions)
    final_actions = np.array(final_actions)

    metrics["mean_residual_action_norm"] = float(np.mean(np.linalg.norm(residual_actions[:,0:3], axis=1)))
    metrics["mean_baseline_action_norm"] = float(np.mean(np.linalg.norm(baseline_actions[:,0:3], axis=1)))
    metrics["mean_final_action_norm"] = float(np.mean(np.linalg.norm(final_actions[:,0:3], axis=1)))
    metrics["initial_distance"] = distances[0]
    metrics["initial_achieved"] = achieved_goals[0]
    metrics["initial_desired"] = desired_goal

    return metrics, frames, residual_actions, baseline_actions, final_actions, achieved_goals


def main():
    env_config = load_config("configs/env.yaml")
    baseline_config = load_config("configs/baseline.yaml")
    residual_config = load_config("configs/residual_sac.yaml")

    env_id = env_config["env_id"]
    best_kp=baseline_config["best_kp"]
    alpha = residual_config["residual_alpha"]

    model_name = "best_model"
    model_label = residual_config["evaluation_model_label"]
    model_path = Path(f"models/residual_sac/{model_label}") / f"{model_name}.zip"
    
    output_path = Path(f"results/metrics/residual_sac/{model_label}/residual_sac_policy_alpha_{alpha}_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run src/train_sac.py first."
        )

    env = make_fetch_env(env_id, render_mode="rgb_array")
    env = ResidualActionWrapper(env, kp=best_kp, alpha=alpha)
    model = SAC.load(model_path, env=env)

    num_episodes = residual_config["num_final_eval_episodes"]
    success_threshold = residual_config["success_threshold"]
    base_seed = env_config["seed"]

    all_metrics = []

    for episode_idx in tqdm(range(num_episodes)):
        record_video = (episode_idx in {0,1,14})

        metrics, frames, residual_actions, baseline_actions, final_actions, achieved_goals = run_one_episode(
            env=env,
            model=model,
            seed=base_seed + episode_idx,# the same seeds are used to enhance reproducebility
            record_video=record_video,
            success_threshold=success_threshold,
        )

        metrics["episode"] = episode_idx
        metrics["policy"] = "sac"
        metrics["env_id"] = env_id
        metrics["kp"] = best_kp
        metrics["residual_alpha"] = alpha
        metrics["model_path"] = str(model_path)

        all_metrics.append(metrics)
        
        episode_label = "000"+str(episode_idx)
        episode_label = episode_label[-3:]

        if record_video:
            save_video(
                frames,
                f"results/videos/residual_sac/{model_label}/residual_sac_policy_alpha_{alpha}_episode_{episode_label}.mp4",
                fps=30,
            )
        # residual_actions, baseline_actions, final_actions, and achieved goals are stored for error analysis
        np.save(f"results/metrics/residual_sac/{model_label}/residual_actions_alpha_{alpha}_episode_{episode_label}.npy", residual_actions)
        np.save(f"results/metrics/residual_sac/{model_label}/baseline_actions_alpha_{alpha}_episode_{episode_label}.npy", baseline_actions)
        np.save(f"results/metrics/residual_sac/{model_label}/final_actions_alpha_{alpha}_episode_{episode_label}.npy", final_actions)
        np.save(f"results/metrics/residual_sac/{model_label}/achieved_goals_alpha_{alpha}_episode_{episode_label}.npy", achieved_goals)

    env.close()

    df = pd.DataFrame(all_metrics)

    df.to_csv(output_path, index=False)

    summary = df.agg(
        {
            "success": "mean",
            "final_distance": "mean",
            "min_distance": "mean",
            "total_return": "mean",
            "mean_action_norm": "mean",
            "action_smoothness": "mean",
            "mean_residual_action_norm": "mean",
            "mean_baseline_action_norm": "mean",
            "mean_final_action_norm": "mean",
        }
    )

    print("\nResidual SAC evaluation complete.")
    print(f"\nalpha = {alpha}")
    print(summary)
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()

