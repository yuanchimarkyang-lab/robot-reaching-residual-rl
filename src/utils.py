"""
Utility Function Definition

This script define three utility function
- goal_distance: it calculate the end effector's current distance to the target
- save_video: it saves the frame as videos.
- summarize_episode: it calculate the summary statistics according to the experiment results.

"""


import numpy as np
import imageio
from pathlib import Path

def goal_distance(obs):
    """
    Objective: to calculate the Euclidean distance between the current achieved goal and the target goal
    Parameters: obs, the object of the observation
    Output: np.float
    """
    distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    return float(distance)


def save_video(frames, output_path, fps=30):
    """
    Objective: to save a list of RGB grames to an mp4 file
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved video to {output_path}")


def summarize_episode(distances, rewards, actions, success_threshold = 0.005):
    """
    Objective: compute simple episode-level metrics.
    """
    actions = np.array(actions)

    #initial_distance = float(distances[0])
    final_distance = float(distances[-1])
    min_distance = float(np.min(distances))
    total_distance = float(np.sum(distances))
    total_return = float(np.sum(rewards))
    mean_action_norm = float(np.mean(np.linalg.norm(actions[:,0:3],axis=1)))
    
    if len(actions)>1:
        action_smoothness = float(np.mean(np.linalg.norm(actions[1:,0:3]-actions[:-1,0:3])))
    else:
        action_smoothness = 0.0


    success = bool(final_distance < success_threshold)

    report = {"final_distance": final_distance,
              "min_distance": min_distance,
              "total_distance": total_distance,
              "total_return": total_return,
              "mean_action_norm": mean_action_norm,
              "num_steps": len(rewards),
              "action_smoothness": action_smoothness,
              "success": success
            }

    return report



