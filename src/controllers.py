import numpy as np

def proportional_controller(obs, kp=5.0):
    """
    Simple goal-reaching proportional controller.

    obs["achieved_goal"] is the current end-effector position.
    obs["desired_goal"] is the target position.

    The FetchReach action has shape (4,):
      action[0:3] = Cartesian displacement command
      action[3]   = gripper command, unused for reaching
    """
    achieved = obs["achieved_goal"]
    desired = obs["desired_goal"]

    delta = desired - achieved

    action = np.zeros(4, dtype=np.float32)
    action[:3] = kp * delta
    action[3] = 0.0

    return np.clip(action, -1.0, 1.0)

