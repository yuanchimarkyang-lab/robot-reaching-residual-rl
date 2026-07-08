# How to achieve accurate and robust control?

## Project Overview
Accurate and robust control under realistic conditions has always been a challenge.
Though it is possible that physics-inspired control policy ideally may achieve accurate control, in many cases, building a realistic physics model could be too costly, let alone that the control is often imperfect. 
Imperfect control could be caused by mismatch between intended action and the actual movement, flawed observation, and environmental noise.
This motivates the adoption of reinforcement learning on robot control, even in simple tasks.

In this project, I focus on a simple task as an example,  `FetchReachDense-v4` from Gymnasium-Robotics. In this task, a simulated Fetch robot arm needs to move its end-effector to a target location. Policies designed and compared first in a noiseless environment; noise is included later in robustness test. 

Currently, a random rollout, a proportional controller, and a SAC-based model, and a residual-SAC based on proportional controller have been implemented and benchmarked.

## Method
### Summary of Policies
| Policy |	Description |
| :----- | :------------|
| Random | Sampled uniformly from the action space |
| Proportional controller | taken directly from the deplacement from the target position, as a simple, physics inspired baseline.|
| SAC | Proposed by a continuous-control policy learned from exploring the environment |
| Residual SAC | Proposed by SAC as a correction added to the proportional controller |

### Random Controller
A simple random controller is implemented to evaluate how random sample might achieve a target as:
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

action = env.action_space.sample() </div>

This is more of a sanity check that such task is not trivial.

### Proportional Controller
A simple, physics-inspired policy named proportional controller that moves the end-effector directly toward the target position is implemented as: 
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

action[:3] = action[:3] = Kp * (desired_goal - achieved_goal)
action[3] = 0</div>

Kp is a variable controlling how large each step needs to be, where Kp $=1,2,5,10,20$ have been tested. A larger Kp might achieve the goal faster with the cost of jerky action, while a too small Kp might not achieve the goal within set number of steps. 

### SAC
A Soft Actor-Critic (SAC) model is implemented based on `stable_baselines3`.
The SAC algorithm is consisted of an actor that predict the best action based on current observation and a pair of critics that predict the Q-values based on the current observation and state.
The soft loss function include an entropy term to encourage exploration.

The SAC model is trained with 150,000 timesteps and evaluated every 5,000 steps. The best model is selected according to the mean episode return on evaluation.

### Residual SAC
Given that the proportional controller can achieve a good baseline but not perfect[^1], it might be possible to further learn to compensate the error using a (residual) SAC, proposed as: 

<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:0px 16px; font-family:monospace; white-space:pre-wrap;">

final_action = proportional_controller + $\alpha$ * residual_action</div>

where $\alpha$ controls the extent of the residual action to the final action. 

Currently, $\alpha=0.3$ is selected, and the residual SAC is trained in a similar manner as SAC.

### Experiment Set-up
| Item | Note | 
| :--- | :---|
|number of evaluation episode | 100 |
| length of each episode | 50 |
| success critetia | $\|d\| < 0.005 \text{ m} = 5 \text{ mm}$ |
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
The perturbation includes three parts: observation noise, action noise, and action level. For observation noise and action noise, we added Gaussian noise on observation or action to evaluate how it would impact the success rate. For action level, we apply a scaling factor to the action to evaluate if the policy is still robust under low-power. 

#### Observation Noise
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:2px 16px; font-family:monospace;white-space:pre-wrap;line-height:1.0;">

$\tilde {s}_t = s_t + \epsilon_t$, where $\tilde {s}_t$ is the noisy observation, $s_t$ is the true observation (state), and $\epsilon_t$ is noise. </div>
<center><img src="./results/metrics/robustness/observation_noise.png" alt="Observation Noise" width="300" style="margin:6px 0 0 0;"></center>

First of all, with an observation noise higher than success threshold ($\sigma \ge 0.005 \text{ m}$), the success rates for all policies drop drastically, indicating that achieving accurate control requires low noise level.
In the regine where the observation noise is about the same size of the sucess threshold the baseline policy ($\sigma = 0.005, 0.01 \text{ m}$), proportional controller seems to be more robust against the observation noise, comparing to SAC and residual SAC. 
The learned policies seem to be more sensitive to small noise, indicating the they might overfit the training data and tend to details that are too small. 
On the other hand, the simplicity of proportional controller seems to make it more robust.   

#### Action Noise
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:2px 16px; font-family:monospace;white-space:pre-wrap;line-height:1.0;">

$\tilde {a}_t = a_t + \epsilon_t$, where $\tilde {a}_t$ is the noisy action, $a_t$ is the action proposed by the policy, and $\epsilon_t$ is noise. </div>
<center><img src="./results/metrics/robustness/action_noise.png" alt="Action Noise" width="300" style="margin:6px 0 0 0;"></center>

Over all, the proportional controller, SAC, and residual SAC are robust against the action noise till $\sigma \ge 0.05 \text{ m}$, suggesting that these policies can correct the error under random noise in action up to this level.
Interestingly, SAC has the higher success score than proportional controller and residual SAC at $\sigma = 0.1 \text{ m}$, suggesting that the probablistic nature of policy might provide additional robustness, in contrast to the proportional controller and the residual SAC where a large part of action is proposed by the proportional controller.  

#### Action Scale
<div style="background: #eeeff0; border:1px solid #a9aaac; border-radius:0px; padding:2px 16px; font-family:monospace;white-space:pre-wrap;line-height:1.0;">

$\bar {a}_t = f* a_t$, where $\bar {a}_t$ is the scaled action $a_t$ is the action proposed by the policy while $f \in (0,1]$ is the scaling factor </div>
<center><img src="./results/metrics/robustness/action_scale.png" alt="Action Noise" width="300" style="margin:6px 0 0 0;"></center>

Over all, the proportional controller, SAC, and residual SAC are robust against the action scale, suggesting that it is the direction of the action that is essential to achieve the goal, not the scale. This is consistent with the finding that the proportional controller can achieve high success rate even with Kp $= 5$.


## Discussion
The cause of the remaining 10% error may includes
* **Mechanical limitation**: the robot arm is not able to reach such target. This might be validated manually on the simulator.
* **Policy Design**: the SAC model does not have enough data for the hard case thus it does not learn. This might be resolved using
    1. a carefully designed loss function to emphasize the fine, final distance such as $\sim \log(d)$. 
    2. a carefully designed scheme to learn the correction on top of the proportional controller, instead of learning from the proportional controller.

## Next


[^1]: The action in FetchReachDense-v4 is interpreted as a Cartesian control command rather than a direct state displacement. Therefore, the realized end-effector motion does not exactly equal the scaled action. This makes the proportional controller a useful but imperfect baseline, motivating residual learning as a way to correct the controller under the simulator dynamics.
