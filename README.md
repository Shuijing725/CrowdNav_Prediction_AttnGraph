# CrowdNav++
This repository contains the codes for our paper titled "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph" in ICRA 2023. 
For more details, please refer to the [project website](https://sites.google.com/view/intention-aware-crowdnav/home) and [arXiv preprint](https://arxiv.org/abs/2203.01821).
For experiment demonstrations, please refer to the [youtube video](https://www.youtube.com/watch?v=nxpxhF019VA).

**[News]**
- Please check out our open-sourced sim2real tutorial [here](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)

## Abstract
We study the problem of safe and intention-aware robot navigation in dense and interactive crowds. 
Most previous reinforcement learning (RL) based methods fail to consider different types of interactions among all agents or ignore the intentions of people, which results in performance degradation. 
In this paper, we propose a novel recurrent graph neural network with attention mechanisms to capture heterogeneous interactions among agents through space and time. 
To encourage longsighted robot behaviors, we infer the intentions of dynamic agents by predicting their future trajectories for several timesteps. 
The predictions are incorporated into a model-free RL framework to prevent the robot from intruding into the intended paths of other agents. 
We demonstrate that our method enables the robot to achieve good navigation performance and non-invasiveness in challenging crowd navigation scenarios. We successfully transfer the policy learned in simulation to a real-world TurtleBot 2i.

<p align="center">
<img src="/figures/open.png" width="450" />
</p>

## Setup
1. In a conda environment or virtual environment with Python 3.x, install the required python package
```
pip install -r requirements.txt
```

2. Install Pytorch 1.12.1 following the instructions [here](https://pytorch.org/get-started/previous-versions/#v1121)

3. Install [OpenAI Baselines](https://github.com/openai/baselines#installation) 
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library


## Overview
This repository is organized in five parts: 
- `crowd_nav/` folder contains configurations and policies used in the simulator.
- `crowd_sim/` folder contains the simulation environment. 
- `gst_updated/` folder contains the code for running inference of a human trajectory predictor, named Gumbel Social Transformer (GST) [2].
- `rl/` contains the code for the RL policy networks, wrappers for the prediction network, and ppo algorithm. 
- `trained_models/` contains some pretrained models provided by us. 

Note that this repository does not include codes for training a trajectory prediction network. Please refer to from [this repo](https://github.com/tedhuang96/gst) instead.

## Run the code
### Training
- Modify the configurations.
  1. Environment configurations: Modify `crowd_nav/configs/config.py`. Especially,
     - Choice of human trajectory predictor: 
       - Set `sim.predict_method = 'inferred'` if a learning-based GST predictor is used [2]. Please also change `pred.model_dir` to be the directory of a trained GST model. We provide two pretrained models [here](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/tree/main/gst_updated/results/).
       - Set `sim.predict_method = 'const_vel'` if constant velocity model is used.
       - Set `sim.predict_method = 'truth'` if ground truth predictor is used.
       - Set `sim.predict_method = 'none'` if you do not want to use future trajectories to change the observation and reward.
     - Randomization of human behaviors: If you want to randomize the ORCA humans, 
       - set `env.randomize_attributes` to True to randomize the preferred velocity and radius of humans;
       - set `humans.random_goal_changing` to True to let humans randomly change goals before they arrive at their original goals.

  2. PPO and network configurations: modify `arguments.py`
     - `env_name` (must be consistent with `sim.predict_method` in `crowd_nav/configs/config.py`): 
        - If you use the GST predictor, set to `CrowdSimPredRealGST-v0`.
        - If you use the ground truth predictor or constant velocity predictor, set to `CrowdSimPred-v0`.
        - If you don't want to use prediction, set to `CrowdSimVarNum-v0`. 
     - `use_self_attn`: human-human attention network will be included if set to True, else there will be no human-human attention.
     - `use_hr_attn`: robot-human attention network will be included if set to True, else there will be no robot-human attention.
- After you change the configurations, run
  ```
  python train.py 
  ```
- The checkpoints and configuration files will be saved to the folder specified by `output_dir` in `arguments.py`.

### Testing
Please modify the test arguments in line 20-33 of `test.py` (**Don't set the argument values in terminal!**), and run   
```
python test.py 
```
Note that the `config.py` and `arguments.py` in the testing folder will be loaded, instead of those in the root directory.  
The testing results are logged in `trained_models/your_output_dir/test/` folder, and are also printed on terminal.  
If you set `visualize=True` in `test.py`, you will be able to see visualizations like this:  
<img src="/figures/visual.gif" width="420" />

#### Test pre-trained models provided by us
| Method                                 | `--model_dir` in test.py               | `--test_model` in test.py |
|----------------------------------------|----------------------------------------|---------------------------|
| Ours without randomized humans         | `trained_models/GST_predictor_no_rand` | `41200.pt`                |
| ORCA without randomized humans         | `trained_models/ORCA_no_rand`          | `00000.pt`                |
| Social force without randomized humans | `trained_models/SF_no_rand`            | `00000.pt`                |
| Ours with randomized humans            | `trained_models/GST_predictor_rand`    | `41665.pt`                |

#### Plot predicted future human positions
To visualize the episodes with predicted human trajectories, as well as saving visualizations to disk, please refer to [save_slides branch](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/tree/save_slides).  
Note that the above visualization and file saving will slow down testing significantly!   
- Set `save_slides=True` in `test.py` and all rendered frames will be saved in a subfolder inside the `trained_models/your_output_dir/social_eval/`.   

### Plot the training curves
```
python plot.py
```
Here are example learning curves of our proposed network model with GST predictor.

<img src="/figures/rewards.png" width="370" /> <img src="/figures/losses.png" width="370" />

## Sim2Real
We are happy to announce that our sim2real tutorial and code are released [here](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)!  
**Note:** This repo only serves as a reference point for the sim2real transfer of crowd navigation. Since there are lots of uncertainties in real-world experiments that may affect performance, we cannot guarantee that it is reproducible on all cases. 

## Disclaimer
1. We only tested our code in Ubuntu with Python 3.6 and Python 3.8. The code may work on other OS or other versions of Python, but we do not have any guarantee.  

2. The performance of our code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)). 
Unfortunately, we do not have time or resources for a thorough hyperparameter search. Thus, if your results are slightly worse than what is claimed in the paper, it is normal. 
To achieve the best performance, we recommend some manual hyperparameter tuning.

## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}

@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517-3524}
}
```

## Credits
Other contributors:  
[Peixin Chang](https://github.com/PeixinC)  
[Zhe Huang](https://github.com/tedhuang96)   
[Neeloy Chakraborty](https://github.com/TheNeeloy)  

Part of the code is based on the following repositories:  

[1] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 3517-3524. (Github: https://github.com/Shuijing725/CrowdNav_DSRNN)  

[2] Z. Huang, R. Li, K. Shin, and K. Driggs-Campbell. "Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1198–1205, 2022. (Github: https://github.com/tedhuang96/gst)  

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.
