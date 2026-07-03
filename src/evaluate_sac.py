from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from tqdm import tqdm

from config import load_config
from env_utils import make_fetch_env
from utils import goal_distance, save_video, summarize_episode


def run_one_episode(
    env,
    model,
    seed=None,
    record_video=False,
    success_threshold=0.05,
):
    obs, info = env.reset(seed=seed)

    frames = []
    distances = []
    achieved_goals = []
    rewards = []
    actions = []
    

    distances.append(goal_distance(obs))
    achieved_goals.append(obs["achieved_goal"])
    desired_goal = obs["desired_goal"]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if record_video:
            frames.append(env.render())

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        distances.append(goal_distance(obs))
        achieved_goals.append(obs["achieved_goal"])
        rewards.append(reward)
        actions.append(action)

    metrics = summarize_episode(
        distances=distances,
        rewards=rewards,
        actions=actions,
        success_threshold=success_threshold,
    )
    
    metrics["initial_distance"] = distances[0]
    metrics["initial_achieved"] = achieved_goals[0]
    metrics["initial_desired"] = desired_goal
    return metrics, frames, actions, achieved_goals


def main():
    env_config = load_config("configs/env.yaml")
    sac_config = load_config("configs/sac.yaml")

    env_id = env_config["env_id"]
    model_name = "best_model"
    model_label = sac_config["evaluation_model_label"]
    model_path = Path(f"models/sac/{model_label}") / f"{model_name}.zip"
    
    output_path = Path(f"results/metrics/sac/{model_label}/sac_policy_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run src/train_sac.py first."
        )

    env = make_fetch_env(env_id, render_mode="rgb_array")

    model = SAC.load(model_path, env=env)

    num_episodes = sac_config["num_final_eval_episodes"]
    success_threshold = sac_config["success_threshold"]
    base_seed = env_config["seed"]

    all_metrics = []

    for episode_idx in tqdm(range(num_episodes)):
        record_video = (episode_idx == 0 or episode_idx == 1 or episode_idx ==14)

        metrics, frames, actions, achieved_goals = run_one_episode(
            env=env,
            model=model,
            seed=base_seed + episode_idx,
            record_video=record_video,
            success_threshold=success_threshold,
        )

        metrics["episode"] = episode_idx
        metrics["policy"] = "sac"
        metrics["env_id"] = env_id
        metrics["model_path"] = str(model_path)

        all_metrics.append(metrics)

        episode_label = "000"+str(episode_idx)
        episode_label = episode_label[-3:]


        if record_video:
            save_video(
                frames,
                f"results/videos/sac/{model_label}/sac_policy_episode_{episode_label}.mp4",
                fps=3,
            )
            actions = np.array(actions)
            np.save(f"results/metrics/sac/{model_label}/actions_sac_policy_episode_{episode_label}.npy",actions)
            achieved_goals = np.array(achieved_goals)
            np.save(f"results/metrics/sac/{model_label}/achieved_goals_sac_policy_episode_{episode_label}.npy", achieved_goals)

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
        }
    )

    print("\nSAC evaluation complete.")
    print(f"\n\tSuccess Threshold: {success_threshold} (m)")
    print(f"\tent_coef = {sac_config['ent_coef']}")
    print(f"\ttarget_entropy = {sac_config['target_entropy']}")
    print(summary)
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()

