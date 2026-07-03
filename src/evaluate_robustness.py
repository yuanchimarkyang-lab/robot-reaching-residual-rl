from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from tqdm import tqdm

from config import load_config
from controllers import proportional_controller
from env_utils import make_fetch_env
from utils import goal_distance, save_video, summarize_episode
from wrappers import (
    ResidualActionWrapper,
    ObservationNoiseWrapper,
    PhysicalActionPerturbationWrapper,
)


def get_wrapper_attr(env, attr_name):
    """
    Search through nested Gym wrappers for an attribute.
    """
    current = env

    while True:
        if hasattr(current, attr_name):
            value = getattr(current, attr_name)
            if value is not None:
                return value

        if not hasattr(current, "env"):
            break

        current = current.env

    return None


def build_eval_env(
    env_id,
    policy_name,
    best_kp,
    residual_alpha,
    obs_noise_sigma=0.0,
    action_noise_sigma=0.0,
    action_scale=1.0,
    obs_noise_keys=None,
    render_mode=None,
):
    """
    Build evaluation environment.

    Wrapper order:

    base MuJoCo env
      -> physical action perturbation
      -> observation noise
      -> residual wrapper, only for residual_sac

    This means:
    - observation noise affects what the policy/controller sees
    - physical action perturbation affects the actual command sent to MuJoCo
    - residual SAC computes its baseline action using noisy observations
    """

    env = make_fetch_env(env_id, render_mode=render_mode)

    env = PhysicalActionPerturbationWrapper(
        env,
        action_noise_sigma=action_noise_sigma,
        action_scale=action_scale,
    )

    env = ObservationNoiseWrapper(
        env,
        sigma=obs_noise_sigma,
        keys=obs_noise_keys,
    )

    if policy_name == "residual_sac":
        env = ResidualActionWrapper(
            env,
            kp=best_kp,
            alpha=residual_alpha,
        )

    return env


def load_models(env_id, best_kp, residual_alpha, env_config, baseline_config, sac_config, residual_config):
    """
    Load SAC and residual SAC models.
    """
    sac_model_path = Path("models/sac") / f"{sac_config['evaluation_model_label']}" / "best_model.zip"
    residual_model_path = Path("models/residual_sac") / f"{residual_config['evaluation_model_label']}"/ "best_model.zip"

    models = {
        "random": None,
        "proportional": None,
        "sac": None,
        "residual_sac": None,
    }

    if sac_model_path.exists():
        print(f"SAC model at: {sac_model_path}")
        dummy_env = build_eval_env(
            env_id=env_id,
            policy_name="sac",
            best_kp=best_kp,
            residual_alpha=residual_alpha,
            render_mode=None,
        )
        models["sac"] = SAC.load(sac_model_path, env=dummy_env)
        dummy_env.close()
    else:
        print(f"Warning: SAC model not found at {sac_model_path}")

    if residual_model_path.exists():
        print(f"residual SAC model at: {residual_model_path}")
        dummy_env = build_eval_env(
            env_id=env_id,
            policy_name="residual_sac",
            best_kp=best_kp,
            residual_alpha=residual_alpha,
            render_mode=None,
        )
        models["residual_sac"] = SAC.load(residual_model_path, env=dummy_env)
        dummy_env.close()
    else:
        print(f"Warning: residual SAC model not found at {residual_model_path}")

    return models


def run_one_episode(
    env,
    policy_name,
    model,
    best_kp,
    seed=None,
    record_video=False,
    success_threshold=0.05,
):
    obs, info = env.reset(seed=seed)

    frames = []
    distances = []
    rewards = []
    actions = []
    achieved_goals = []

    distances.append(goal_distance(obs))
    achieved_goals.append(obs["achieved_goal"])
    

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if record_video:
            frames.append(env.render())

        if policy_name == "random":
            action = env.action_space.sample()

        elif policy_name == "proportional":
            action = proportional_controller(obs, kp=best_kp)

        elif policy_name in ["sac", "residual_sac"]:
            if model is None:
                raise ValueError(f"Model for policy {policy_name} was not loaded.")
            action, _ = model.predict(obs, deterministic=True)

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs, reward, terminated, truncated, info = env.step(action)

        true_obs = info.get("true_obs", obs)

        distances.append(goal_distance(true_obs))
        achieved_goals.append(true_obs["achieved_goal"])
        rewards.append(reward)

        physical_action = get_wrapper_attr(env, "last_physical_action")
        final_action = get_wrapper_attr(env, "last_final_action")

        if physical_action is not None:
            actions.append(physical_action)
        elif final_action is not None:
            actions.append(final_action)
        else:
            actions.append(action)

    metrics = summarize_episode(
        distances=distances,
        rewards=rewards,
        actions=actions,
        success_threshold=success_threshold,
    )

    return metrics, frames, actions, achieved_goals


def evaluate_setting(
    policy_name,
    model,
    env_id,
    best_kp,
    residual_alpha,
    obs_noise_sigma,
    action_noise_sigma,
    action_scale,
    obs_noise_keys,
    num_episodes,
    base_seed,
    success_threshold,
    record_video=False,
    video_tag="",
):
    render_mode = "rgb_array" if record_video else None

    env = build_eval_env(
        env_id=env_id,
        policy_name=policy_name,
        best_kp=best_kp,
        residual_alpha=residual_alpha,
        obs_noise_sigma=obs_noise_sigma,
        action_noise_sigma=action_noise_sigma,
        action_scale=action_scale,
        obs_noise_keys=obs_noise_keys,
        render_mode=render_mode,
    )

    all_metrics = []

    for episode_idx in tqdm(range(num_episodes), leave=False):
        should_record = record_video and episode_idx == 0

        metrics, frames, actions, achieved_goals = run_one_episode(
            env=env,
            policy_name=policy_name,
            model=model,
            best_kp=best_kp,
            seed=base_seed + episode_idx,
            record_video=should_record,
            success_threshold=success_threshold,
        )

        metrics["episode"] = episode_idx
        metrics["policy"] = policy_name
        metrics["obs_noise_sigma"] = obs_noise_sigma
        metrics["action_noise_sigma"] = action_noise_sigma
        metrics["action_scale"] = action_scale

        all_metrics.append(metrics)
        
        episodes_label = "000" + str(episode_idx)
        episodes_label = episodes_label[-3:]

        if should_record:
            save_video(
                frames,
                f"results/videos/robustness_{video_tag}_{policy_name}.mp4",
                fps=30,
            )
        np.save("results/metrics/robustness/actions_{video_tag}_{policy_name}_episodes_{episodes_label}.npy", actions)
        np.save("results/metrics/robustness/achieved_goals_{video_tag}_{policy_name}_episodes_{episodes_label}.npy", achieved_goals)

    env.close()

    return all_metrics


def main():
    env_config = load_config("configs/env.yaml")
    baseline_config = load_config("configs/baseline.yaml")
    sac_config = load_config("configs/sac.yaml")
    residual_config = load_config("configs/residual_sac.yaml")
    robustness_config = load_config("configs/robustness.yaml")

    env_id = env_config["env_id"]
    base_seed = env_config["seed"]

    best_kp = baseline_config["best_kp"]
    residual_alpha = residual_config["residual_alpha"]

    num_episodes = robustness_config["num_eval_episodes"]
    success_threshold = robustness_config["success_threshold"]
    policies = robustness_config["policies"]
    obs_noise_keys = robustness_config["obs_noise_keys"]

    record_videos = robustness_config.get("record_videos", False)
    
    output_path = Path("results/metrics/robustness/robustness_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = load_models(
        env_id=env_id,
        best_kp=best_kp,
        residual_alpha=residual_alpha,
        env_config=env_config,
        baseline_config=baseline_config,
        sac_config=sac_config,
        residual_config=residual_config,
    )

    all_results = []

    experiments = []

    for sigma in robustness_config["observation_noise_levels"]:
        experiments.append(
            {
                "experiment": "observation_noise",
                "level": sigma,
                "obs_noise_sigma": sigma,
                "action_noise_sigma": 0.0,
                "action_scale": 1.0,
            }
        )

    for sigma in robustness_config["action_noise_levels"]:
        experiments.append(
            {
                "experiment": "action_noise",
                "level": sigma,
                "obs_noise_sigma": 0.0,
                "action_noise_sigma": sigma,
                "action_scale": 1.0,
            }
        )

    for scale in robustness_config["action_scale_levels"]:
        experiments.append(
            {
                "experiment": "action_scale",
                "level": scale,
                "obs_noise_sigma": 0.0,
                "action_noise_sigma": 0.0,
                "action_scale": scale,
            }
        )

    for exp in experiments:
        print(
            f"\nExperiment={exp['experiment']}, "
            f"level={exp['level']}, "
            f"obs_noise={exp['obs_noise_sigma']}, "
            f"action_noise={exp['action_noise_sigma']}, "
            f"action_scale={exp['action_scale']}"
        )

        for policy_name in policies:
            print(f"  Evaluating policy: {policy_name}")

            model = models.get(policy_name)

            # Skip learned policies if model is missing
            if policy_name in ["sac", "residual_sac"] and model is None:
                print(f"  Skipping {policy_name}: model not available.")
                continue

            should_record_video = (
                record_videos
                and exp["experiment"] in ["observation_noise", "action_noise"]
                and exp["level"] != 0.0
            )

            metrics = evaluate_setting(
                policy_name=policy_name,
                model=model,
                env_id=env_id,
                best_kp=best_kp,
                residual_alpha=residual_alpha,
                obs_noise_sigma=exp["obs_noise_sigma"],
                action_noise_sigma=exp["action_noise_sigma"],
                action_scale=exp["action_scale"],
                obs_noise_keys=obs_noise_keys,
                num_episodes=num_episodes,
                base_seed=base_seed,
                success_threshold=success_threshold,
                record_video=should_record_video,
                video_tag=f"{exp['experiment']}_{exp['level']}",
            )

            for row in metrics:
                row["experiment"] = exp["experiment"]
                row["level"] = exp["level"]
                row["env_id"] = env_id
                row["best_kp"] = best_kp
                row["residual_alpha"] = residual_alpha

            all_results.extend(metrics)

    df = pd.DataFrame(all_results)

    df.to_csv(output_path, index=False)

    summary = df.groupby(["experiment", "level", "policy"]).agg(
        success_rate=("success", "mean"),
        mean_final_distance=("final_distance", "mean"),
        median_final_distance=("final_distance", "median"),
        mean_return=("total_return", "mean"),
        mean_action_norm=("mean_action_norm", "mean"),
        mean_action_smoothness=("action_smoothness", "mean"),
    ).reset_index()

    summary_path = Path("results/metrics/robustness_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nRobustness evaluation complete.")
    print(summary)
    print(f"\nSaved full metrics to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

