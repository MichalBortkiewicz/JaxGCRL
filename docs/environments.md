We provide 8 blazingly fast goal-conditioned environments based on [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [BRAX](https://github.com/google/brax) and jitted framework for 
quick experimentation with goal-conditioned self-supervised reinforcement learning.


| Environment  |                         Env name                         |  Code  |                             Documentation                             |
|:-------------|:--------------------------------------------------------:|:------:|:---------------------------------------------------------------------:|
| Reacher      |                        `reacher`                         | [LINK] |   [link](https://gymnasium.farama.org/environments/mujoco/reacher/)   |
| Half Cheetah |                        `cheetah`                         | [LINK] | [link](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) |
| Pusher       |             `pusher_easy` <br> `pusher_hard`             | [LINK] |    [link](https://gymnasium.farama.org/environments/mujoco/pusher/)   |
| Ant          |                          `ant`                           | [LINK] |     [link](https://gymnasium.farama.org/environments/mujoco/ant/)     |
| Ant Maze     | `ant_u_maze` <br> `ant_big_maze` <br> `ant_hardest_maze` | [LINK] |
| Ant Soccer   |                        `ant_ball`                        | [LINK] |
| Ant Push     |                        `ant_push`                        | [LINK] |
| Humanoid     |                        `humanoid`                        | [LINK] |  [link](https://gymnasium.farama.org/environments/mujoco/humanoid/)   |


### Adding new environments
Each environment implementation has 2 main parts: an XML file and a Python file. 

The XML file contains information about geometries, placements, properties, and movements of objects in the environment. Depending on the Brax pipeline used, the XML file may vary slightly, but generally, it should follow [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html). Since all environments are vectorized and compiled with JAX, the information in [MJX guide](https://mujoco.readthedocs.io/en/stable/mjx.html) should also be taken into consideration, particularly the [feature parity](https://mujoco.readthedocs.io/en/stable/mjx.html#feature-parity) section and [performance tuning](https://mujoco.readthedocs.io/en/stable/mjx.html#performance-tuning) section.
!!! note annotate "XML files"
    In our experience XML files that worked with standard MuJoCo require some tuning for MJX. In particular, the number of solver iterations should be carefully adjusted, so that the environment is fast but still stable.


The Python file contains the logic of the environment, a description of how the environment is initialized, restored, and how one environment step looks. The class describing the environment should inherit from BRAX's [`PipelineEnv`](https://github.com/google/brax/blob/f43727eeebf21c031faf861ee00e98919c892140/brax/envs/base.py#L75) class. All environment logic should be JIT-able with JAX, which requires some care in using certain Python instructions like `if` and `for`. The observation returned by the `step` function of the environment should be a state of the environment concatenated with the current environment goal. 

To use the new environment it should be added to the `create_env` function in `utils.py`.