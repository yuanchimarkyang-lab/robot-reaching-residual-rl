# Accurate and Robust Robot Reaching, a project based on FetchReachDense-V4 in MuJoCo
##  To-do:
* add a script to copy config and training evaluation, and wrapper script into storage.
* maybe consider a more smooth loss function such as -np.log(d+\epsilon) 
## Project Overview
Accurately and robustly control robot has been a challenge.
The reason behind imperfect control includes the mismatch between intended action and the actual movement as well as imperfect observation.
If such control error is persistent even in a noiseless environment, it might be originated in the robot design and might be counterbalanced by a carefully designed policy.
On the other hand, since the environment is inevitably noisy, the policy needs to be robust under such policy.

In this project, I focus on a simple task,  `FetchReachDense-v4` from Gymnasium-Robotics, where a simulated Fetch robot arm needs to move its end-effector to a target location. 
Policies for such task are designed and compared first in a noiseless environment; noise is included later in robustness test. 

Currently, a random rollout, a proportional controller, and a SAC-based model, and a residual-SAC based on proportional controller have been implemented and benchmarked.

## Method
### Summary of Policies
| Policy |	Description |
| :----- | :------------|
| Random | Samples actions uniformly from the environment action space |
| Proportional controller | Uses the vector from achieved goal to desired goal as a Cartesian control signal|
| SAC | Learns a continuous-control policy from environment interaction |
| Residual SAC | Learns a correction added to the proportional controller |

### Random Controller
A simple random controller is implemented to evaluate how random sample might achieve a target as:
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

action = env.action_space.sample() </div>


### Proportional Controller
A physics-inspired proportional controller that moves the end-effector directly toward the target position is implemented as: 
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

action[:3] = action[:3] = Kp * (desired_goal - achieved_goal)
action[3] = 0</div>

Kp is a variable controlling how large each step needs to be. A larger Kp might achieve the goal faster with the cost of jerky action, while a too small Kp might not achieve the goal within set number of steps. 

### SAC
A Soft Actor-Critic (SAC) model is implemented based on `stable_baselines3`.
The SAC algorithm is consisted of an actor that predict the best action based on current observation and a pair of critics that predict the Q-values based on the current observation and state.
The loss function is soft because it also include an entropy term to encourage exploration.

The SAC model is trained with 150,000 timesteps, evaluated every 5,000 steps, and best model selected according to the mean episode return on evaluation.

### Residual SAC
Given that the proportional controller can achieve a good baseline, it might be possible to further learn to compensate the error with SAC. 
Therefore, a residual SAC is implemented as:

<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

final_action = proportional_controller + $\alpha$ * residual_action</div>

where $\alpha$ controls the extent of the residual action to the final action. 

Currently, $\alpha=0.3$ is selected, and the residual SAC is also trained with 150,000 timesteps, evaluated every 5,000 steps, and best model selected according to the mean episode return on evaluation.

### Experiment Set-up
| Item | Note | 
| :--- | :---|
|number of evaluation episode | 100 |
| length of each episode | 50 |
| success critetia | $\|d\| < 0.005 m = 5 mm$ |
| reward | $-\sum_{i=1}^N \|d_i\|^2$ |

where $d = $ achieved_goal - target_goal, *the distance between the end effector and the target position*.

#### Note
* The action in FetchReachDense-v4 is interpreted as a Cartesian control command rather than a direct state displacement. Therefore, the realized end-effector motion does not exactly equal the scaled action. This makes the proportional controller a useful but imperfect baseline, motivating residual learning as a way to correct the controller under the simulator dynamics.
* The success criteria is 10 times stricker than the default (0.05 m). Such success criteria might render the default reward function ( $\propto d$ ) ineffective because the differences would be too small close to the success, affecting learning efficiency 

## Result
### Without Perturbation
| Policy | Success Rate | Mean Episode Return | Final Distance (m) |
| :--- | :---: | :---: | :---: |
| Random | 0% | -9.54 $\pm$ 2.86 | 0.217 $\pm$ 0.098 |
| Proportional Controller |  90% | -0.277 $\pm$ 0.318 | 0.0018 $\pm$ 0.0056 |
| SAC | 89% | -0.332 $\pm$ 0.309 | 0.0031 $\pm$ 0.0052 |
| Residual SAC |  90% | -0.331 $\pm$ 0.298 | 0.0031 $\pm$ 0.0052 |

#### Random Rollout
The random rollout, unsurprisingly, results in 0% success rate and large mean episode return and final distance. This shows that this task requires a carefully designed policy and can not be achieved purely by luck.


#### Proportional Controller
The proportional controller exhibit a strong baseline, with final distance below the success criteria (5 mm). 
However, the 10% unsuccess rate and large deviation in the mean episode return and final distance indicates that there are cases that it can not resolve. 

In fact, error analysis on one episode shows that the gap in the 3rd direction can not be overcome by simply calling for actions in the 3rd direction and is directly related to the imperfect control. This suggests that a learned policy on the imperfection might be able to further improve the performance. 

#### SAC
The SAC model seems to exhibit a strong performance, with sucess rate, mean episode return, and the final distance approaching those of the proportional controller.
This suggests that SAC might have learned that actions similar to that of the proportional controller. 

However, since the hard cases only take about 10%, the SAC model seems to learn the majority, easy cases while the scarce, hard cases left unlearned.

#### Residual SAC
The Residual SAC is designed to learn the residual action that the SAC model to perform to compensate what the proportional controller can not do, i.e., to overcome the imperfect control.
However, the results show that the residual SAC likely also learn how the proportional controller perform as well. 

A new design is required. 


### With Perturbation


## Discussion
The cause of the remaining 10% error may includes
* **Mechanical limitation**: the robot arm is not able to reach such target. This might be validated manually on the simulator.
* **Policy Design**: the SAC model does not have enough data for the hard case thus it does not learn. This might be resolved using
    1. a carefully designed loss function to emphasize the fine, final distance, such as $\sim \log(d)$. 
    2. a carefully designed scheme to learn the correction on top of the proportional controller, instead of learning from the proportional controller.

## Next

