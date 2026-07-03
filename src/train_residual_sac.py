from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import load_config
from env_utils import make_fetch_env
from wrappers import ResidualActionWrapper, CustomThresholdWrapper

from datetime import datetime, timezone, timedelta

def make_residual_env(env_id,kp,alpha):
    env = make_fetch_env(env_id, render_mode=None)
    env = ResidualActionWrapper(env, kp=kp, alpha=alpha)
    env = Monitor(env)
    return env



def main():
    env_config=load_config("configs/env.yaml")
    baseline_config=load_config("configs/baseline.yaml")
    residual_config=load_config("configs/residual_sac.yaml")

    env_id=env_config["env_id"]
    best_kp=baseline_config["best_kp"]
    alpha=residual_config["residual_alpha"]
    model_name = f"{residual_config['model_name']}"
    time_label = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    model_label = f"{alpha}_{residual_config['ent_coef']}_{residual_config['target_entropy']}_{residual_config['total_timesteps']}_{time_label}"

    model_dir=Path(f"models/residual_sac/{model_label}")
    log_dir=Path(f"logs/residual_sca/{model_label}")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_residual_env(env_id, kp=best_kp, alpha = alpha)
    eval_env = make_residual_env(env_id, kp=best_kp, alpha = alpha)
    eval_env = CustomThresholdWrapper(eval_env)

    eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=residual_config["eval_freq"],
            n_eval_episodes=residual_config["n_eval_episodes"],
            deterministic=True,
            render=False,
            )


    model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=residual_config["learning_rate"],
            buffer_size=residual_config["buffer_size"],
            batch_size=residual_config["batch_size"],
            learning_starts=residual_config["learning_starts"],
            ent_coef=residual_config["ent_coef"],
            target_entropy=residual_config["target_entropy"],
            gamma=residual_config["gamma"],
            tau=residual_config["tau"],
            train_freq=residual_config["train_freq"],
            gradient_steps=residual_config["gradient_steps"],
            verbose=1,
            seed=env_config["seed"]
            )

    print(f"Training residual Sac on {env_id}")
    print(f"Baseline kp: {best_kp}")
    print(f"Residual alpha: {alpha}")
    print(f"Total timesteps: {residual_config['total_timesteps']}")

    model.learn(
            total_timesteps=residual_config["total_timesteps"],
            callback=eval_callback,
            progress_bar=True,
    )

    final_model_path=model_dir/f"{model_name}.zip"
    model.save(final_model_path)

    print(f"\nSaved final SAC model to:{final_model_path}")
    print(f"\n\talpha = {alpha}")
    print(f"\tent_coef = {residual_config['ent_coef']}")
    print(f"\ttarget_entropy = {residual_config['target_entropy']}")
    print(f"Best model, if available, saved under:{model_dir/'best_model'}")

    train_env.close()
    eval_env.close()

if __name__=="__main__":
    main()
