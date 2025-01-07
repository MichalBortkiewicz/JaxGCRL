<h1 align="center"> JaxGCRL</h1>


<p align="center">
    <a href= "https://arxiv.org/abs/2408.11052">
        <img src="https://img.shields.io/badge/arXiv-2311.10090-b31b1b.svg" /></a>
    <a href= "https://github.com/MichalBortkiewicz/JaxGCRL/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://michalbortkiewicz.github.io/JaxGCRL/">
        <img src="https://img.shields.io/badge/docs-green" /></a>
</p>


<p align="center"><img src="imgs/grid.png" width=80%></p>

[**Installation**](#Installation) | [**Quick Start**](#start) | [**Environments**](#envs) | [**Baselines**](#baselines) | [**Citation**](#cite)
---

## Accelerating Goal-Conditioned RL Algorithms and Research

We provide blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.

- **Blazing Fast Training** - Train 10 million environment steps in 10 
  minutes on a single GPU, up to 22$\times$ faster than prior implementations.
- **Comprehensive Benchmarking** - Includes 10+ diverse environments and multiple pre-implemented baselines for out-of-the-box evaluation.
- **Modular Implementation** - Designed for clarity and scalability, 
  allowing for easy modification of algorithms.


## Installation 📂
The entire process of installing the benchmark is just one step using the conda `environment.yml` file.
```bash
conda env create -f environment.yml
```

<h3 name="start" id="start">Quick Start 🚀 </h3>

To check whether installation worked, run a test experiment using `./scripts/train.sh` file:

```bash
chmod +x ./scripts/train.sh; ./scripts/train.sh
```
> [!NOTE]  
> If you haven't configured yet [`wandb`](https://wandb.ai/site), you might be prompted to log in.

To run experiments of interest, change `scripts/train.sh`; descriptions of flags are in `utils.py:create_parser()`. Common flags you may want to change:
- **env=...**: replace "ant" with any environment name. See `utils.py:create_env()` for names.
- Removing **--log_wandb**: omits logging, if you don't want to use a wandb account.
- **--num_timesteps**: shorter or longer runs.
- **--num_envs**: based on how many environments your GPU memory allows.
- **--contrastive_loss_fn, --energy_fn, --h_dim, --n_hidden, etc.**: algorithmic and architectural changes.

### Environment Interaction

This section demonstrates how to interact with the environment using the `reset` and `step` functions. The environment returns a state object, which is a dataclass containing the following fields:

`state.pipeline_state`: current, internal state of the environment\
`state.obs`: current observation\
`state.done`: flag indicating if the agent reached the goal\
`state.metrics`: agent performance metrics\
`state.info`: additional info

The following code demonstrates how to interact with the environment:

```python
import jax
from utils import create_env

key = jax.random.PRNGKey(0)

# Initialize the environment
env = create_env('ant')

# Use JIT compilation to make environment's reset and step functions execute faster
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

NUM_STEPS = 1000

# Reset the environment and obtain the initial state
state = jit_env_reset(key)

# Simulate the environment for a fixed number of steps
for _ in range(NUM_STEPS):
    # Generate a random action
    key, key_act = jax.random.split(key, 2)
    random_action = jax.random.uniform(key_act, shape=(8,), minval=-1, maxval=1)
    
    # Perform an environment step with the generated action
    state = jit_env_step(state, random_action)
```

### Wandb support 📈
We highly recommend using Wandb for tracking and visualizing your results ([Wandb support](##wandb-support)). Enable Wandb logging with the `--log_wandb` flag. Additionally, you can organize experiments with the following flags:
- `--project_name`
- `--group_name`
- `--exp_name`

Logging to W&B happens when the `--log_wandb` flag is used when it's not used, metrics are logging to CSV file.

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

<h2 name="envs" id="envs">Environments 🌎</h2>

We currently support a number of continuous control environments:
- Locomotion: Half-Cheetah, Ant, Humanoid
- Locomotion + task: AntMaze, AntBall (AntSoccer), AntPush, HumanoidMaze
- Simple arm: Reacher, Pusher, Pusher 2-object
- Manipulation: Reach, Grasp, Push (easy/hard), Binpick (easy/hard)


| Environment   |                                Env name                                |                      Code                       |
|:--------------|:----------------------------------------------------------------------:|:-----------------------------------------------:|
| Reacher       |                               `reacher`                                |            [link](./envs/reacher.py)            |
| Half Cheetah  |                               `cheetah`                                |         [link](./envs/half_cheetah.py)          |
| Pusher        |                    `pusher_easy` <br> `pusher_hard`                    |            [link](./envs/pusher.py)             |
| Ant           |                                 `ant`                                  |              [link](./envs/ant.py)              |
| Ant Maze      |        `ant_u_maze` <br> `ant_big_maze` <br> `ant_hardest_maze`        |           [link](./envs/ant_maze.py)            |
| Ant Soccer    |                               `ant_ball`                               |           [link](./envs/ant_ball.py)            |
| Ant Push      |                               `ant_push`                               |           [link](./envs/ant_push.py)            |
| Humanoid      |                               `humanoid`                               |           [link](./envs/humanoid.py)            |
| Humanoid Maze | `humanoid_u_maze` <br> `humanoid_big_maze` <br>`humanoid_hardest_maze` |         [link](./envs/humanoid_maze.py)         |
| Arm Reach     |                              `arm_reach`                               |    [link](./envs/manipulation/arm_reach.py)     |
| Arm Grasp     |                              `arm_grasp`                               |    [link](./envs/manipulation/arm_grasp.py)     |
| Arm Push      |                  `arm_push_easy` <br> `arm_push_hard`                  |  [link](./envs/manipulation/arm_push_easy.py)   |
| Arm Binpick   |             `arm_binpick_easy` <br> `arm_binpick_hard`             | [link](./envs/manipulation/arm_binpick_easy.py) |

To add new environments: add an XML to `envs/assets`, add a python environment file in `envs`, and register the environment name in `utils.py`.

<h2 name="baselines" id="baselines">Baselines 🤖</h2>

We currently support following algorithms:

| Algorithm                                     | How to run                             | Code                                     |
|-----------------------------------------------|----------------------------------------|------------------------------------------|
| [CRL](https://arxiv.org/abs/2206.07568)       | `python training.py ...`               | [link](./src/train.py)                   |
| [SAC](https://arxiv.org/abs/1801.01290)       | `python training_sac.py ...`           | [link](./src/baselines/sac.py)           |
| [SAC + HER](https://arxiv.org/abs/1707.01495) | `python training_sac.py ... --use_her` | [link](./src/baselines/sac.py)           |
| [TD3](https://arxiv.org/pdf/1802.09477)       | `python training_td3.py ...`           | [link](./src/baselines/td3/td3_train.py) |
| [TD3 + HER](https://arxiv.org/abs/1707.01495) | `python training_td3.py ... --use_her` | [link](./src/baselines/td3/td3_train.py) |
| [PPO](https://arxiv.org/abs/1707.06347)       | `python training_ppo.py ...`           | [link](./src/baselines/ppo.py)           |


## Code Structure 📝
We summarize the most important elements of the code structure, for users wanting to understand the implementation specifics or modify the code:

<pre><code>
├── <b>src:</b> Algorithm code (training, network, replay buffer, etc.)
│   ├── <b>train.py:</b> Main file. Defines energy functions + losses, and networks. Collects trajectories, trains networks, runs evaluations.
│   ├── <b>replay_buffer.py:</b> Contains replay buffer, including logic for state, action, and goal sampling for training.
│   └── <b>evaluator.py:</b> Runs evaluation and collects metrics.
├── <b>envs:</b> Environments (python files and XMLs)
│   ├── <b>ant.py, humanoid.py, ...:</b> Most environments are here
│   ├── <b>assets:</b> Contains XMLs for environments
│   └── <b>manipulation:</b> Contains all manipulation environments
├── <b>scripts/train.sh:</b> Modify to choose environment and hyperparameters
├── <b>utils.py:</b> Logic for script argument processing, rendering, environment names, etc.
└── <b>training.py:</b> Interface file that processes script arguments, calls train.py, initializes wandb, etc.
</code></pre>

To modify the architecture: modify `networks.py`.


## Contributing 🏗️
Help us build JaxGCRL into the best possible tool for the GCRL community.
Reach out and start contributing or just add an Issue/PR!

- [x] Add Franka robot arm environments. [Done by SimpleGeometry]
- [x] Get around 70% success rate on Ant Big Maze task. [Done by RajGhugare19]
- [ ] Add more complex versions of Ant Sokoban.
- [ ] Integrate environments: 
    - [ ] Overcooked 
    - [ ] Hanabi
    - [ ] Rubik's cube
    - [ ] Sokoban
 

<h2 name="cite" id="cite">Citing JaxGCRL 📜 </h2>
If you use JaxGCRL in your work, please cite us as follows:

```
@article{bortkiewicz2024accelerating,
  title   = {Accelerating Goal-Conditioned RL Algorithms and Research},
  author  = {Michał Bortkiewicz and Władek Pałucki and Vivek Myers and Tadeusz Dziarmaga and Tomasz Arczewski and Łukasz Kuciński and Benjamin Eysenbach},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2408.11052}
}
```

## Questions ❓
If you have any questions, comments, or suggestions, please reach out to Michał Bortkiewicz ([michalbortkiewicz8@gmail.com](michalbortkiewicz8@gmail.com)).


## See Also 🙌
There are a number of other libraries which inspired this work, we encourage you to take a look!

JAX-native algorithms:
- [Mava](https://github.com/instadeepai/Mava): JAX implementations of IPPO and MAPPO, two popular MARL algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
- [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.
- [JaxIRL](https://github.com/FLAIROx/jaxirl?tab=readme-ov-file): JAX implementation of algorithms for inverse reinforcement learning.

JAX-native environments:
- [Gymnax](https://github.com/RobertTLange/gymnax): Implementations of classic RL tasks including classic control, bsuite and MinAtar.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse set of environments ranging from simple games to NP-hard combinatorial problems.
- [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- [Brax](https://github.com/google/brax): A fully differentiable physics engine written in JAX, features continuous control tasks.
- [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid): Meta-RL gridworld environments inspired by XLand and MiniGrid.
- [Craftax](https://github.com/MichaelTMatthews/Craftax): (Crafter + NetHack) in JAX.
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): Multi-agent RL in Jax.
