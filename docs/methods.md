## Contrastive RL
The main algorithm implemented in this repository is Contrastive Reinforcement Learning (CRL), described in [Contrastive Learning as Goal-Conditioned
Reinforcement Learning](https://arxiv.org/pdf/2206.07568). This algorithm learns a critic function without rewards by contrasting states sampled from a future of a given trajectory (positive samples) with states sampled from a random trajectory (negative samples). 
We implemented a number of modifications and improvements to this algorithm. Among those, the most important ones are:

- Choice of the energy function, currently we have: `norm`, `l2`, `dot`, and `cosine` energy functions.
- Choice of the contrastive loss, currently we have: `binary_nce`, `fwd_infonce`, `bwd_infonce`, and `sym_infonce` losses.
- `logsumexp_penalty`- this is a regularizing term applied to contrastive loss which usually improves performance and is mathematically necessary for the `fwd_infonce` loss.

These implementations can be found and modified in `agents/crl/crl.py` file.

## Other baselines
To easily compare CRL to other well-known RL algorithms we have implemented several other baselines including PPO, SAC, and TD3, based on implementations in [Brax](https://github.com/google/brax). For SAC and TD3 it is additionally possible to enable `--use_her` flag to take advantage of the Hindsight Experience Replay buffer, which can improve performance in sparse reward setting. 

An advantage of the goal-conditioned RL setting is that rewards are not assumed. Nonetheless, for methods that can use reward, most environments can provide both sparse and dense rewards, controlled with the `--use_dense_reward` flag.

## Adding new methods and algorithms
The primary purpose of our work is to enable easy and rapid research on self-supervised goal-conditioned reinforcement learning. Thus adding new losses, and energy functions, or changing other aspects of the algorithm can be easily done, by modifying `agents/crl/losses.py` and/or `agents/crl/networks.py`, which are easily readable and accessible.


### Adding new contrastive objective
For instance, to register a new contrastive objective ("`your_loss`") for CRL, you need to:
1. Register the new function `your_loss` in the `contrastive_loss_fn` function (`agents/crl/losses.py`):
```python
...
if name == "your_loss":
    critic_loss = ...
...
```
2. Run training with your new contrastive objective:
```shell
python run.py crl --contrastive_loss_fn "your_loss"
```
### Using a custom model architecture.

To integrate a custom model architecture for CRL algorithm, you need to define and register your architecture within the `agents/crl/networks.py` file: 
1. Critic: extend or modify the `MLP` class or create a new model used for contrastive embeddings. 
2. Actor: provide appropriate `make_policy_network` function that defines actor architecture.    

