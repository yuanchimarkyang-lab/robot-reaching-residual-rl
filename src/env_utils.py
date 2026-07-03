import gymnasium as gym
import gymnasium_robotics


def make_fetch_env(env_id, render_mode=None):
    """
    Create a Gymnasium-Robotics Fetch environment.

    For training:
        render_mode=None

    For video recording:
        render_mode="rgb_array"
    """
    gym.register_envs(gymnasium_robotics)

    if render_mode is None:
        return gym.make(env_id)

    return gym.make(env_id, render_mode=render_mode)

