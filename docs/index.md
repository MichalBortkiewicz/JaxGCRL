# <span style="color: orange;">JaxGCRL</span>


<p align="center"><img src="imgs/grid.png" width=80%></p>

JaxGCRL is a high-performance library and benchmark for self-supervised goal-conditioned reinforcement learning. 
Leveraging efficient GPU acceleration, the framework enables researchers to train agents for millions of environment 
steps within minutes on a single GPU.

- **Blazing Fast Training** - Train 10 million environment steps in 10 
  minutes on a single GPU, up to 22$\times$ faster than prior implementations.
- **Comprehensive Benchmarking** - Includes 10+ diverse environments and multiple pre-implemented baselines for out-of-the-box evaluation.
- **Modular Implementation** - Designed for clarity and scalability, 
  allowing for easy modification of algorithms.


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



## Paper: Accelerating Goal-Conditioned RL Algorithms and Research
<p align="center">
  <img src="imgs/teaser.jpg" width=100% /> 
</p>
<p align="center">
Training CRL on Ant environment for 10M steps takes only ~10 minutes on 1 Nvidia V100. 
</p>

**Abstract:** Self-supervision has the potential to transform reinforcement learning (RL), paralleling the breakthroughs it has enabled in other areas of machine learning. While self-supervised learning in other domains aims to find patterns in a fixed dataset, self-supervised goal-conditioned reinforcement learning (GCRL) agents discover new behaviors by learning from the goals achieved during unstructured interaction with the environment. However, these methods have failed to see similar success, both due to a lack of data from slow environment simulations as well as a lack of stable algorithms. We take a step toward addressing both of these issues by releasing a high-performance codebase and benchmark (JaxGCRL) for self-supervised GCRL, enabling researchers to train agents for millions of environment steps in minutes on a single GPU. By utilizing GPU-accelerated replay buffers, environments, and a stable contrastive RL algorithm, we reduce training time by up to $\mathbf{22\times}$ . Additionally, we assess key design choices in contrastive RL, identifying those that most effectively stabilize and enhance training performance. With this approach, we provide a foundation for future research in self-supervised GCRL, enabling researchers to quickly iterate on new ideas and evaluate them in diverse and challenging environments.

