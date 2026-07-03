import gymnasium as gym
import gymnasium_robotics

from config import load_config

def main():
    gym.register_envs(gymnasium_robotics)
    
    config = load_config()

    env=gym.make(
            config["env_id"], 
            render_mode=config["render_mode"])

    obs, info = env.reset(seed=config["seed"])

    print("Environment created successfully.")
    print("Environment ID:", config["env_id"])
    print("Observation type:", type(obs))
    print("Observation keys:", obs.keys())

    print("\nObservation shapes:")
    for key, value in obs.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        print(f"  {key}: ", value)

    print("\nAction space:")
    print(env.action_space)
    print("Action shape:", env.action_space.shape)
    print("Action low:", env.action_space.low)
    print("Action high:", env.action_space.high)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    print("\nOne random step succeeded.")
    print("Action:", action)
    print("Dispalcement: ", action*0.03)
    print("next_obs:", next_obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    frame = env.render()
    print("\nRendered frame shape:", frame.shape)

    env.close()


if __name__=="__main__":
    main()
