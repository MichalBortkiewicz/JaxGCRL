## Implemented algorithms
### Contrastive RL
The main algorithm implemented in this repository is Contrastive Reinforcement Learning (CRL), first described in a paper [Contrastive Learning as Goal-Conditioned
Reinforcement Learning](https://arxiv.org/pdf/2206.07568). This Algorithm learns a critic function, without rewards, by contrasting states sampled from a future of a given trajectory (positive samples) with states sampled from a random trajectory (negative samples). 
While the general method is always the same we implemented a number of modifications and improvements. Among those, the most important ones are:
* Choice of the energy function, currently we have implemented: `l1`, `l2`, `l2_no_sqrt`, `dot`, and `cos` energy functions.
* Choice of the contrastive loss, currently we have implemented: `binary`, `infonce`, `infonce_backward`, `symmetric_infonce`, `flat_infonce`, `flat_infonce_backward`, `forward_backward`, `dpo`. `ipo`, and `sppo` losses.
* `logsumex_penalty`- this is a regularizing term applied to contrastive loss, that we have found improves training in most settings.
All of the above, and more, can be found and modified  in `src/losses.py` file.
### Other baselines
To easily compare CRL to other well-known RL algorithms we have implemented several other baselines including PPO, SAC, and TD3, based on implementations in [Brax](https://github.com/google/brax). For SAC and TD3 it is additionally possible to enable `--use_her` flag to take advantage of the Hindsight Experience Replay buffer, which can improve performance in sparse reward setting. 

While reward shaping was not our main priority, since CRL algorithm doesn't use reward, each environment can provide both sparse and dense rewards, by using `--use_dense_reward` flag.

### Adding new methods and algorithms
The primary purpose of our work is to enable easy and rapid research on Goal-Conditioned Refinorcement Learning. Thus adding new losses, and energy functions, or changing other aspects of the algorithm can be easily done, by modifying `src/losses.py` and/or `src/networks.py`, which are easily readable and accessible.

Algorithms, that differ from CRL (or one of the other implemented baselines) in a more fundamental way (e.g. non-standard replay buffer, not relaying on actor and critic as a main paradigm) can also be implemented, but will usually require modification of `src/train.py`, which requires some technical knowledge on JAX, especially how JIT mechanism works.
