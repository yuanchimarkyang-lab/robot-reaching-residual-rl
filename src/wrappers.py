"""
Definition for Wrappers that interact with the environment differently.

This scripts define
- ResidualActionWrapper: the wrapper for residual_sac as a correction to proportional controller
- ObservationNoiseWrapper: the wrapper to add observation noise.
- PhysicalActionPerturbationWrapper: the wrapper to add action noise and action scale
- CustomThresholdWrapper: the wrapper to modify the threshold to 0.005 on eval during SAC and residual SAC training.
"""

import gymnasium as gym
import numpy as np

from utils import goal_distance
from controllers import proportional_controller

class ResidualActionWrapper(gym.Wrapper):
    """
    Environment wrapper for residual reinforcement learning.

    The agent outputs a residual action.
    The environment receives:

        final_action = baseline_action + alpha * residual_action

    where baseline_action is computed from the proportional controller.
    """
    def __init__(self,env,kp=20.0,alpha=0.3):
        super().__init__(env)

        self.kp=kp
        self.alpha = alpha

        self.current_obs = None

        self.last_baseline_action = None
        self.last_residual_action = None
        self.last_final_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs

        self.last_baseline_action = None
        self.last_residual_action = None
        self.last_final_action = None

        return obs, info
    
    def step(self, residual_action):
        if self.current_obs is None:
            raise RuntimeError("Environment must be reset before calling step()")

        residual_action = np.array(residual_action, dtype=np.float32)

        baseline_action = proportional_controller(self.current_obs, kp=self.kp)
        final_action = baseline_action + self.alpha * residual_action
        final_action = np.clip(final_action, self.action_space.low, self.action_space.high).astype(np.float32)

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        self.last_baseline_action = baseline_action
        self.last_residual_action = residual_action
        self.last_final_action = final_action

        self.current_obs = obs

        distance = goal_distance(obs)
        #reward = -np.log(distance) # newly added reward to encourage small distance.

        return obs, reward, terminated, truncated, info


class ObservationNoiseWrapper(gym.Wrapper):
    """
    Adds Gaussian noise to selected observation dictionary keys.

    Important:
    - The policy/controller sees the noisy observation.
    - Evaluation metrics should use the true observation stored in info["true_obs"].

    """

    def __init__(self, env, sigma=0.0, keys=None):
        super().__init__(env)
        self.sigma=sigma
        self.keys = keys or ["observation", "achieved_goal"]
        self.rng=np.random.default_rng()


    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self.rng=np.random.default_rng(seed+12345)

        obs, info = self.env.reset(**kwargs)

        true_obs = self._copy_obs(obs)
        noisy_obs = self._add_noise(obs)

        info = dict(info)
        info["true_obs"]=true_obs

        return noisy_obs, info

    def step(self,action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        true_obs = self._copy_obs(obs)
        noisy_obs = self._add_noise(obs)

        info = dict(info)
        info["true_obs"] = true_obs

        return noisy_obs, reward, terminated, truncated, info

    def _copy_obs(self,obs):
        return {key:value.copy() for key, value in obs.items()}

    def _add_noise(self, obs):
        noisy_obs=self._copy_obs(obs)

        if self.sigma<=0:
            return noisy_obs

        for key in self.keys:
            if key in noisy_obs:
                noise = self.rng.normal(loc=0.0, scale=self.sigma, size=noisy_obs[key].shape)
                noisy_obs[key]=noisy_obs[key]+noise
        return noisy_obs


class PhysicalActionPerturbationWrapper(gym.ActionWrapper):
    """
    Applies physical action perturbation before the action reaches the base environment.

    This simulates actuator noise or actuator scaling.

        physical_action = action_scale * action + Gaussian noise

    The final command is clipped to the environment action bounds.
    """

    def __init__(self,env,action_noise_sigma=0.0,action_scale=1.0):
        super().__init__(env)

        self.action_noise_sigma = action_noise_sigma
        self.action_scale = action_scale

        self.rng = np.random.default_rng()

        self.last_raw_action = None
        self.last_physical_action = None

    
    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self.rng = np.random.default_rng(seed + 54321)

        self.last_raw_action = None
        self.last_physical_action = None

        return self.env.reset(**kwargs)


    def action(self,action):
        raw_action=np.asarray(action,dtype=np.float32)

        if self.action_noise_sigma>0:
            noise = self.rng.normal(
                loc=0.0,
                scale=self.action_noise_sigma,
                size=raw_action.shape,
            )
        else:
            noise = 0.0

        physical_action = self.action_scale * raw_action + noise

        physical_action = np.clip(physical_action, self.action_space.low, self.action_space.high).astype(np.float32)

        self.last_raw_action = raw_action
        self.last_physical_action = physical_action

        return physical_action



class CustomThresholdWrapper(gym.ActionWrapper):
    """
    This wrapper would pass the customized threshold for evaluation environment
    """

    def __init__(self,env,threshold=0.005):
        super().__init__(env)

        self.threshold = threshold

    def step(self,action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        info = dict(info)
        info["is_success"] = float(distance < self.threshold)

        return obs, reward, terminated, truncated, info


