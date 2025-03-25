<h1 align="center"> JaxGCRL</h1>


<p align="center">
    <a href= "https://arxiv.org/abs/2408.11052">
        <img src="https://img.shields.io/badge/arXiv-2311.10090-b31b1b.svg" /></a>
    <a href="https://pypi.org/project/jaxgcrl/">
        <img src="https://img.shields.io/pypi/v/jaxgcrl" /></a>
    <a href= "https://github.com/MichalBortkiewicz/JaxGCRL/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
    <a href= "https://michalbortkiewicz.github.io/JaxGCRL/">
        <img src="https://img.shields.io/badge/docs-green" /></a>
    <a href= "https://michalbortkiewicz.github.io/JaxGCRL/">
        <img src="https://img.shields.io/badge/website-purple" /></a>
</p>


<p align="center"><img src="https://raw.githubusercontent.com/MichalBortkiewicz/JaxGCRL/master/imgs/grid_transparent.png" width=85%></p>

<center>

[**Installation**](#Installation) | [**Quick Start**](#start) | [**Environments**](#envs) | [**Baselines**](#baselines) | [**Citation**](#cite)

</center>

<br/>

## Accelerating Goal-Conditioned RL Algorithms and Research

We provide blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.

- **Blazingly Fast Training** - Train 10 million environment steps in 10 
  minutes on a single GPU, up to $22\times$ faster than prior implementations.
- **Comprehensive Benchmarking** - Includes 10+ diverse environments and multiple pre-implemented baselines for out-of-the-box evaluation.
- **Modular Implementation** - Designed for clarity and scalability, 
  allowing for easy modification of algorithms.


## Installation 📂

#### Editable Install (Recommended)

After cloning the repository, run one of the following commands.

With GPU on Linux:
```bash
pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

With CPU on Mac:
```bash
export SDKROOT="$(xcrun --show-sdk-path)" # may be needed to build brax dependencies
pip install -e . 
```

#### PyPI

The package is also available on PyPI:
```bash
pip install jaxgcrl -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

<h3 name="start" id="start">Quick Start 🚀 </h3>

To verify the installation, run a test experiment:
```bash
jaxgcrl crl --env ant
```

The `jaxgcrl` command is equivalent to invoking `python run.py` with the same arguments

> [!NOTE]  
> If you haven't yet configured [`wandb`](https://wandb.ai/site), you may be prompted to log in.

See `scripts/train.sh` for an example config. 
A description of the available agents can be generated with `jaxgcrl --help`.
Available configs can be listed with `jaxgcrl {crl,ppo,sac,td3} --help`.
Common flags you may want to change include:
- **env=...**: replace "ant" with any environment name. See `jaxgcrl/utils/env.py` for a list of available environments.
- Removing **--log_wandb**: omits logging, if you don't want to use a wandb account.
- **--total_env_steps**: shorter or longer runs.
- **--num_envs**: based on how many environments your GPU memory allows.
- **--contrastive_loss_fn, --energy_fn, --h_dim, --n_hidden, etc.**: algorithmic and architectural changes.

### Environment Interaction

Environments can be controlled with the `reset` and `step` functions. These methods return a state object, which is a dataclass containing the following fields:

`state.pipeline_state`: current, internal state of the environment\
`state.obs`: current observation\
`state.done`: flag indicating if the agent reached the goal\
`state.metrics`: agent performance metrics\
`state.info`: additional info

The following code demonstrates how to interact with the environment:

```python
import jax
from utils.env import create_env

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
We strongly recommend using Wandb for tracking and visualizing results ([Wandb support](##wandb-support)). Enable Wandb logging with the `--log_wandb` flag. The following flags are also available to organize experiments:
- `--project_name`
- `--group_name`
- `--exp_name`

The `--log_wandb` flag logs metrics to Wandb. By default, metrics are logged to a CSV.

1. Run example [`sweep`](https://docs.wandb.ai/guides/sweeps):
```bash
wandb sweep --project example_sweep ./scripts/sweep.yml
```
2. Then run `wandb agent` with :
```
wandb agent <previous_command_output>
```

We also render videos of the learned policies as `wandb` artifacts. 

<p align="center">
  <img src="https://raw.githubusercontent.com/MichalBortkiewicz/JaxGCRL/master/imgs/wandb.png" width=55% />
  <img src="https://raw.githubusercontent.com/MichalBortkiewicz/JaxGCRL/master/imgs/push.gif" width=40%  /> 
</p>

<h2 name="envs" id="envs">Environments 🌎</h2>

We currently support a variety of continuous control environments:
- Locomotion: Half-Cheetah, Ant, Humanoid
- Locomotion + task: AntMaze, AntBall (AntSoccer), AntPush, HumanoidMaze
- Simple arm: Reacher, Pusher, Pusher 2-object
- Manipulation: Reach, Grasp, Push (easy/hard), Binpick (easy/hard)


| Environment     | Env name                                                                 | Code                                              |
| :-------------- | :----------------------------------------------------------------------: | :-----------------------------------------------: |
| Reacher         | `reacher`                                                                | [link](./jaxgcrl/envs/reacher.py)                         |
| Half Cheetah    | `cheetah`                                                                | [link](./jaxgcrl/envs/half_cheetah.py)                    |
| Pusher          | `pusher_easy` <br> `pusher_hard`                                         | [link](./jaxgcrl/envs/pusher.py)                          |
| Ant             | `ant`                                                                    | [link](./jaxgcrl/envs/ant.py)                             |
| Ant Maze        | `ant_u_maze` <br> `ant_big_maze` <br> `ant_hardest_maze`                 | [link](./jaxgcrl/envs/ant_maze.py)                        |
| Ant Soccer      | `ant_ball`                                                               | [link](./jaxgcrl/envs/ant_ball.py)                        |
| Ant Push        | `ant_push`                                                               | [link](./jaxgcrl/envs/ant_push.py)                        |
| Humanoid        | `humanoid`                                                               | [link](./jaxgcrl/envs/humanoid.py)                        |
| Humanoid Maze   | `humanoid_u_maze` <br> `humanoid_big_maze` <br>`humanoid_hardest_maze`   | [link](./jaxgcrl/envs/humanoid_maze.py)                   |
| Arm Reach       | `arm_reach`                                                              | [link](./jaxgcrl/envs/manipulation/arm_reach.py)          |
| Arm Grasp       | `arm_grasp`                                                              | [link](./jaxgcrl/envs/manipulation/arm_grasp.py)          |
| Arm Push        | `arm_push_easy` <br> `arm_push_hard`                                     | [link](./jaxgcrl/envs/manipulation/arm_push_easy.py)      |
| Arm Binpick     | `arm_binpick_easy` <br> `arm_binpick_hard`                               | [link](./jaxgcrl/envs/manipulation/arm_binpick_easy.py)   |

To add new environments: add an XML to `envs/assets`, add a python environment file in `envs`, and register the environment name in `utils.py`.

<h2 name="baselines" id="baselines">Baselines 🤖</h2>

We currently support following algorithms:

| Algorithm                                       | How to run                               | Code                                       |
| ----------------------------------------------- | ---------------------------------------- | ------------------------------------------ |
| [CRL](https://arxiv.org/abs/2206.07568)         | `python run.py crl ...`                 | [link](./jaxgcrl/agents/crl/)                      |
| [PPO](https://arxiv.org/abs/1707.06347)         | `python run.py ppo ...`                 | [link](./jaxgcrl/agents/ppo/)                      |
| [SAC](https://arxiv.org/abs/1801.01290)         | `python run.py sac ...`                 | [link](./jaxgcrl/agents/sac/)                      |
| [SAC + HER](https://arxiv.org/abs/1707.01495)   | `python run.py sac ... --use_her`       | [link](./jaxgcrl/agents/sac/)                      |
| [TD3](https://arxiv.org/pdf/1802.09477)         | `python run.py td3 ...`                 | [link](./jaxgcrl/agents/td3/)                      |
| [TD3 + HER](https://arxiv.org/abs/1707.01495)   | `python run.py td3 ... --use_her`       | [link](./jaxgcrl/agents/td3/)                      |


## Code Structure 📝

The core structure of the codebase is as follows:

<pre><code>
<b>run.py:</b> Takes the name of an agent and runs with the specified configs.
<b>agents/</b>
├── <b>agents/</b>
│   ├── <b>crl/</b> 
│   │   ├── <b>crl.py</b> CRL algorithm 
│   │   ├── <b>losses.py</b> contrastive losses and energy functions
│   │   └── <b>networks.py</b> CRL network architectures
│   ├── <b>ppo/</b> 
│   │   └── <b>ppo.py</b> PPO algorithm 
│   ├── <b>sac/</b> 
│   │   ├── <b>sac.py</b> SAC algorithm
│   │   └── <b>networks.py</b> SAC network architectures
│   └── <b>td3/</b> 
│       ├── <b>td3.py</b> TD3 algorithm
│       ├── <b>losses.py</b> TD3 loss functions
│       └── <b>networks.py</b> TD3 network architectures
├── <b>utils/</b>
│   ├── <b>config.py</b> Base run configs
│   ├── <b>env.py</b> Logic for rendering and environment initialization
│   ├── <b>replay_buffer.py:</b> Contains replay buffer, including logic for state, action, and goal sampling for training.
│   └── <b>evaluator.py:</b> Runs evaluation and collects metrics.
├── <b>envs/</b>
│   ├── <b>ant.py, humanoid.py, ...:</b> Most environments are here.
│   ├── <b>assets:</b> Contains XMLs for environments.
│   └── <b>manipulation:</b> Contains all manipulation environments.
└── <b>scripts/train.sh:</b> Modify to choose environment and hyperparameters.
</code></pre>

The architecture can be adjusted in `networks.py`.


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

To run tests (make sure you have access to a GPU):
```bash
python -m pytest 
```

<h2 name="cite" id="cite">Citing JaxGCRL 📜 </h2>
If you use JaxGCRL in your work, please cite us as follows:

```bibtex
@inproceedings{bortkiewicz2025accelerating,
    author    = {Bortkiewicz, Micha\l{} and Pa\l{}ucki, W\l{}adek and Myers, Vivek and
                 Dziarmaga, Tadeusz and Arczewski, Tomasz and Kuci\'{n}ski, \L{}ukasz and
                 Eysenbach, Benjamin},
    booktitle = {{International Conference} on {Learning Representations}},
    title     = {{Accelerating Goal-Conditioned RL Algorithms} and {Research}},
    url       = {https://arxiv.org/pdf/2408.11052},
    year      = {2025},
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

