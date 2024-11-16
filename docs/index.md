# <span style="color: orange;">JaxGCRL</span>



## New CRL implementation and Benchmark
<p align="center">
  <img src="imgs/teaser.jpg" width=100% /> 
</p>

<p align="center">
Training CRL on Ant environment for 10M steps takes only ~10 minutes on 1 Nvidia V100. 
</p>

We provide 8 blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) and jitted framework for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.  


<p align="center"><img src="imgs/grid.png" width=80%></p>



## Structure of the Code
The codebase is organized into several key files and directories. Below is an overview of the structure and most important files:

```
├── training.py
├── src 
│ ├── train.py
│ ├── networks.py
│ ├── losses.py
│ └── ...
├── envs
│ └── ...
├── utils.py
└── ...
```
**`training.py`** - The main entry point for running training. It initializes essential components such as the environment, configuration, logging, and starts the training loop.

**`src/train.py`** - Implements the training loop for a GCRL agent.

**`src/networks.py`** - Defines the neural network architectures.

**`src/losses.py`** - Provides customizable loss functions.

**`envs`** - Contains implementations of various environments.

**`utils.py`** - Provides utility functions and classes to support training, including argument parsing, environment creation, configuration management, and metrics logging.


## Environments

This section lists the available environments in the repository, along with the environment names and the corresponding code links

| Environment | Env name | Code |
| :- | :-: | :-: |
| Reacher |  `reacher`  |  [link](./envs/reacher.py)  |
| Half Cheetah | `cheetah` | [link](./envs/half_cheetah.py)  |
| Pusher | `pusher_easy` <br> `pusher_hard`|  [link](./envs/pusher.py)  |
| Ant |  `ant`  |  [link](./envs/ant.py)  |
| Ant Maze |  `ant_u_maze` <br> `ant_big_maze` <br> `ant_hardest_maze`  |  [link](./envs/ant_maze.py)  |
| Ant Soccer |  `ant_ball`  |  [link](./envs/ant_ball.py)  |
| Ant Push |  `ant_push`  |  [link](./envs/ant_push.py)  |
| Humanoid | `humanoid`|  [link](./envs/humanoid.py)  |


## Paper: Accelerating Goal-Conditioned RL Algorithms and Research


**Abstract:** Self-supervision has the potential to transform reinforcement learning (RL), paralleling the breakthroughs it has enabled in other areas of machine learning. While self-supervised learning in other domains aims to find patterns in a fixed dataset, self-supervised goal-conditioned reinforcement learning (GCRL) agents discover new behaviors by learning from the goals achieved during unstructured interaction with the environment. However, these methods have failed to see similar success, both due to a lack of data from slow environment simulations as well as a lack of stable algorithms. We take a step toward addressing both of these issues by releasing a high-performance codebase and benchmark (JaxGCRL) for self-supervised GCRL, enabling researchers to train agents for millions of environment steps in minutes on a single GPU. By utilizing GPU-accelerated replay buffers, environments, and a stable contrastive RL algorithm, we reduce training time by up to $\mathbf{22\times}$ . Additionally, we assess key design choices in contrastive RL, identifying those that most effectively stabilize and enhance training performance. With this approach, we provide a foundation for future research in self-supervised GCRL, enabling researchers to quickly iterate on new ideas and evaluate them in diverse and challenging environments.

