"""
Training SAC

This script train SAC model with set-up given by configs/sac.yaml

"""

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from datetime import datetime, timezone, timedelta

from config import load_config
from env_utils import make_fetch_env
from wrappers import CustomThresholdWrapper

def main():
    env_config=load_config("configs/env.yaml")
    sac_config=load_config("configs/sac.yaml")

    env_id=env_config["env_id"]
    model_name = sac_config["model_name"]
    time_label = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    model_label = f"{sac_config['ent_coef']}_{sac_config['target_entropy']}_{sac_config['total_timesteps']}_{time_label}"

    model_dir=Path(f"models/sac/{model_label}")
    log_dir=Path(f"logs/sac/{model_label}")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_fetch_env(env_id, render_mode=None)
    train_env = Monitor(train_env)

    eval_env = make_fetch_env(env_id, render_mode=None)
    eval_env = CustomThresholdWrapper(eval_env) # the success threshold is modified to 0.005
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=sac_config["eval_freq"],
            n_eval_episodes=sac_config["n_eval_episodes"],
            deterministic=True,
            render=False,
            )


    model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=sac_config["learning_rate"],
            buffer_size=sac_config["buffer_size"],
            batch_size=sac_config["batch_size"],
            learning_starts=sac_config["learning_starts"],
            ent_coef=sac_config["ent_coef"],
            target_entropy=sac_config["target_entropy"],
            gamma=sac_config["gamma"],
            tau=sac_config["tau"],
            train_freq=sac_config["train_freq"],
            gradient_steps=sac_config["gradient_steps"],
            verbose=1,
            seed=env_config["seed"]
            )

    print(f"Training Sac on {env_id}")
    print(f"Total timesteps: {sac_config['total_timesteps']}")

    model.learn(
            total_timesteps=sac_config["total_timesteps"],
            callback=eval_callback,
            progress_bar=True,
    )

    final_model_path=model_dir/f"{model_name}.zip"
    model.save(final_model_path)

    print(f"\nSaved final SAC model to:{final_model_path}")
    print(f"\n\tent_coef = {sac_config['ent_coef']}")
    print(f"\ttarget_entropy = {sac_config['target_entropy']}")
    print(f"Best model, if svailabel, saved under:{model_dir/'best_model'}")

    train_env.close()
    eval_env.close()

if __name__=="__main__":
    main()
