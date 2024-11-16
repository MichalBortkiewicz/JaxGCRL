# Accelerating Goal-Conditioned RL Algorithms and Research 


<p align="center"><img src="imgs/grid.png" width=80%></p>




**Abstract:** Self-supervision has the potential to transform reinforcement learning (RL), paralleling the breakthroughs it has enabled in other areas of machine learning. While self-supervised learning in other domains aims to find patterns in a fixed dataset, self-supervised goal-conditioned reinforcement learning (GCRL) agents discover new behaviors by learning from the goals achieved during unstructured interaction with the environment. However, these methods have failed to see similar success, both due to a lack of data from slow environment simulations as well as a lack of stable algorithms. We take a step toward addressing both of these issues by releasing a high-performance codebase and benchmark (JaxGCRL) for self-supervised GCRL, enabling researchers to train agents for millions of environment steps in minutes on a single GPU. By utilizing GPU-accelerated replay buffers, environments, and a stable contrastive RL algorithm, we reduce training time by up to $22\times$
. Additionally, we assess key design choices in contrastive RL, identifying those that most effectively stabilize and enhance training performance. With this approach, we provide a foundation for future research in self-supervised GCRL, enabling researchers to quickly iterate on new ideas and evaluate them in diverse and challenging environments.




## Installation
The entire process of installing the benchmark is just one step using the conda `environment.yml` file.
```bash
conda env create -f environment.yml
```

To check whether installation worked, run a test experiment using `./scripts/train.sh` file:

```bash
chmod +x ./scripts/train.sh; ./scripts/train.sh
```
> [!NOTE]  
> If you haven't configured yet [`wandb`](https://wandb.ai/site), you might be prompted to log in.

## New CRL implementation and Benchmark
<p align="center">
  <img src="imgs/teaser.jpg" width=100% /> 
</p>

<p align="center">
Training CRL on Ant environment for 10M steps takes only ~10 minutes on 1 Nvidia V100. 
</p>

We provide 8 blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) and jitted framework for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.  

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

## Experiments
JaxGCRL is highly flexible in terms of parameters, allowing for a wide range of experimental setups. To run a basic experiment, you can start with:
```
python training.py --env_name ant
```
For a complete list of environments, refer to the [environments section](##Environments) or [source code](./utils.py#L66).

One of JaxGCRL's key features is its ability to run parallel environments for data collection. If your GPU has limited memory, you can reduce the number of parallel environments with the following parameter:

```
python training.py --env_name ant --num_envs 16 --batch_size 16
```

> [!NOTE] 
> `num_envs * (episode_length - 1)` must be divisible by `batch_size` due to the way data is stored in replay buffer.

You can customize the neural network architecture by adjusting the number of hidden layers (`n_hidden`), the width of hidden layers (`h_dim`) and the representation dimension (`repr_dim`):
```
python training.py --env_name ant --h_dim 128 --n_hidden 3 --repr_dim 32
```

JaxGCRL supports various built-in energy functions and contrastive loss functions. For example, to use an L2 penalty with the InfoNCE-backward contrastive loss, you can run:
```
python training.py --env_name ant --energy_fn l2 --contrastive_loss_fn infonce_backward
```
For a full list of available energy functions and contrastive losses, see: [ [energy functions](./src/losses.py#L91) | [contrastive losses](./src/losses.py#L145) ]

JaxGCRL offers many other useful parameters, such as `num_timesteps`, `batch_size`, `episode_length`. For a complete list of parameters, their descriptions, and default values, refer to [link](./utils.py#L24).

To execute multiple experiments, you can use a bash script. We highly recommend using Wandb for tracking and visualizing your results ([Wandb support](##wandb-support)). Enable Wandb logging with the `--log_wandb` flag. Additionally, you can organize experiments with the following flags:
- `--project_name`
- `--group_name`
- `--exp_name`



For example, the script below runs experiments to test the performance of different contrastive losses:
```bash
for c_loss in binary infonce flatnce fb dp; do
	for seed in 1 2 3 4 5; do
		python training.py --project_name crl --group_name contrastive_loss_experiments \
		--exp_name c_loss --seed ${seed} --num_timesteps 10000000 --num_evals 50 \
		--batch_size 256 --num_envs 512 --discounting 0.99 --action_repeat 1 \
		--episode_length 1000 --unroll_length 62  --min_replay_size 1000 \
		--max_replay_size 10000 --contrastive_loss_fn ${c_loss} --energy_fn l2 \
		--multiplier_num_sgd_steps 1 --log_wandb
	done
done
```

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


## Wandb support
All of the metric runs are logged into `wandb`. We recommend using it as a tool for running sweep over hyperparameters.

1. Run exemplary [`sweep`](https://docs.wandb.ai/guides/sweeps):
```bash
wandb sweep --project exemplary_sweep ./scripts/sweep.yml
```
2. Then run wandb agent with :
```
wandb agent <previous_command_output>
```


Besides logging the metrics, we also render final policy to `wandb` artifacts. 

<p align="center">
  <img src="imgs/wandb.png" width=55% />
  <img src="imgs/push.gif" width=40%  /> 
</p>
