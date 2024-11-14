<p align="center"><img src="imgs/grid.png" width=80%></p>

# JaxGCRL: A New CRL implementation and Benchmark

<p align="center">
Michał Bortkiewicz,  Władek Pałucki,  Vivek Myers,
</p>

<p align="center">
Tadeusz Dziarmaga,  Tomasz Arczewski,
</p>

<p align="center">
Łukasz Kuciński,  Benjamin Eysenbach
</p>


<p style="text-align: center;">
    Paper: <a href="https://arxiv.org/abs/2408.11052" target="_blank">Accelerating Goal-Conditioned RL Algorithms and Research</a>
</p>

<p align="center">
  <img src="imgs/teaser.jpg" width=100% /> 
</p>

<p align="center">
Training CRL on the Ant environment for 10M steps takes only ~10 minutes on 1 Nvidia V100. 
</p>

We provide blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.

## Supported Environments

We currently support a number of continuous control environments:
- Locomotion: Half-Cheetah, Ant, Humanoid
- Locomotion + task: AntMaze, AntBall (AntSoccer), AntPush, HumanoidMaze
- Simple arm: Reacher, Pusher, Pusher 2-object
- Manipulation: Reach, Grasp, Push (easy/hard), Binpick (easy/hard)

### Environment Docs
Information about most environments can be found in the paper, and information about the manipulation environments can be
found in the markdown file in envs/manipulation.

Documentation is somewhat sparse, so when in doubt look at the environment files and XMLs for the exact implementation. We hope 
to improve the docs in the future (please submit a PR if you'd like to help)!

## Basic Usage
After cloning, setup the conda environment:
```bash
conda env create -f environment.yml
```

Then, try out a basic experiment in Ant with the training script:

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

## Code Structure
We summarize the most important elements of the code structure, for users wanting to understand the implementation specifics or modify the code:

<pre><code>
├── <b>src:</b> Algorithm code (training, network, replay buffer, etc.)
│   ├── <b>train.py:</b> Main file. Collects trajectories, trains networks, runs evaluations.
│   ├── <b>losses.py:</b> Contains energy functions, and actor, critic, and alpha losses.
│   ├── <b>networks.py:</b> Contains network definitions for policy, and encoders for the critic.
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

To add new environments: add an XML to `envs/assets`, add a python environment file in `envs`, and register the environment name in `utils.py`.

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

In addition, you can find exemplary plotting utils for data downloaded by `wandb` api in notebooks.

## Contributing
Help us build JaxGCRL into the best possible tool for the GCRL community.
Reach out and start contributing or just add an Issue/PR!

- [x] Add Franka robot arm environments. [Done by SimpleGeometry]
- [ ] Add more complex versions of Ant Sokoban.
- [ ] Get around 70% success rate on Ant Big Maze task.
- [ ] Integrate environments: 
    - [ ] Overcooked 
    - [ ] Hanabi
    - [ ] Rubik's cube
    - [ ] Sokoban
 
## Questions?
If you have any questions, comments, or suggestions, please reach out to Michał Bortkiewicz ([michalbortkiewicz8@gmail.com](michalbortkiewicz8@gmail.com))


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


## Citation
```
@article{bortkiewicz2024accelerating,
  title   = {Accelerating Goal-Conditioned RL Algorithms and Research},
  author  = {Michał Bortkiewicz and Władek Pałucki and Vivek Myers and Tadeusz Dziarmaga and Tomasz Arczewski and Łukasz Kuciński and Benjamin Eysenbach},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2408.11052}
}
```