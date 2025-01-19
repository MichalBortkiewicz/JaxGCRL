import functools
import time
from typing import Callable, Optional, NamedTuple

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
from jax import numpy as jnp
import optax
from absl import logging
import brax
from brax import envs
from brax.training import  types, pmap  #,distribution
from brax.training.replay_buffers_test import jit_wrap

from envs.wrappers import TrajectoryIdWrapper
from src import distribution
from src.evaluator import CrlEvaluator
from src.replay_buffer import Transition, TrajectoryUniformSamplingQueue

Metrics = types.Metrics
Env = envs.Env
State = envs.State
_PMAP_AXIS_NAME = "i"


def loss_and_pgrad(loss_fn: Callable[..., float], pmap_axis_name: Optional[str], has_aux: bool = False):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state, grads

    return f



# The SAEncoder, GoalEncoder, and Actor all use the same function. Output size for SA/Goal encoders should be representation size, and for Actor should be 2 * action_size.
# To keep parity with the existing architecture, by default we only use one residual block of depth 2, hence effectively not using the residual connections.
class Net(nn.Module):
    """
    MLP with residual connections: residual blocks have $block_size layers. Uses swish activation, optionally uses layernorm.
    """
    output_size: int
    width: int = 1024
    num_blocks: int = 1
    block_size: int = 2
    use_ln: bool = True
    @nn.compact
    def __call__(self, x):
        lecun_uniform = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = lambda x: nn.LayerNorm()(x) if self.use_ln else (lambda x: x)

        for i in range(self.num_blocks*self.block_size):
            x = nn.Dense(self.width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)

            if True:
                if i == 0:
                    skip = x
                if i > 0 and i % self.block_size == 0:
                    x = x + skip
                    skip = x
                
        # Last layer mapping to representation dimension
        x = nn.Dense(self.output_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x

# The brax version of this does not take in the actor and action_distribution arguments; before we pass it to brax evaluator or return it from train(), we do a partial application.
def make_policy(actor, parametric_action_distribution, params, deterministic=False):
    def policy(obs, key_sample):
        logits = actor.apply(params, obs)
        if deterministic:
            action = parametric_action_distribution.mode(logits)
        else:
            action = parametric_action_distribution.sample(logits, key_sample)
        extras = {}
        return action, extras
    return policy

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState

def _init_training_state(key, actor, sa_encoder, g_encoder, state_dim, goal_dim, action_dim, actor_lr, critic_lr, alpha_lr, num_local_devices_to_use):
    """
    Initializes the training state for a contrastive reinforcement learning model. This function sets up the initial states for various components including the policy
    network, CRL networks, and optimizers. All parameters are initialized and replicated across the specified
    number of local devices.

    Args:
        key (PRNGKey): A pseudorandom number generator key used for initializing various network parameters.
        actor (function): The flax function for the actor network.
        sa_encoder (function): The flax function for the state-action encoder network.
        g_encoder (function): The flax function for the goal encoder network.
        state_dim (int): The dimension of the observations from the environment, not including goals.
        goal_dim (int): The dimension of the goals in the environment.
        action_dim (int): The dimension of the action that the actor should pass into the environment in env.step.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the state-action and goal encoder networks.
        alpha_lr (float): The learning rate for the entropy coefficient (alpha).
        num_local_devices_to_use (int): The number of local devices to utilize for training.

    Returns:
        TrainingState: An initialized TrainingState object that contains the initial states of the
        policy network parameters, CRL critic parameters, their respective optimizer states, alpha parameter,
        and normalization parameters.
    """
    actor_key, sa_key, g_key = jax.random.split(key, 3)
    
    # Actor and entropy coefficient
    actor_params = actor.init(actor_key, jnp.ones([1, state_dim + goal_dim]))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))
    log_alpha = {"log_alpha": jnp.array(0.0)}
    alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(learning_rate=alpha_lr))

    # Critic
    sa_encoder_params = sa_encoder.init(sa_key, jnp.ones([1, state_dim + action_dim]))
    g_encoder_params = g_encoder.init(g_key, jnp.ones([1, goal_dim]))
    critic_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
    critic_state = TrainState.create(apply_fn=None, params=critic_params, tx=optax.adam(learning_rate=critic_lr))

    # Put everything together into TrainingState
    training_state = TrainingState(env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()), 
                                   actor_state=actor_state, critic_state=critic_state, alpha_state=alpha_state)
    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_local_devices_to_use])
    return training_state



def compute_metrics(logits, sa_repr, g_repr, l2_loss, l_align, l_unif):
    I = jnp.eye(logits.shape[0])
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
    if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1) ** 2
    else:
        logsumexp = jax.nn.logsumexp(logits, axis=1) ** 2

    sa_repr_l2 = jnp.sqrt(jnp.sum(sa_repr**2, axis=-1))
    g_repr_l2 = jnp.sqrt(jnp.sum(g_repr**2, axis=-1))

    # l_align_log = -jnp.mean(jnp.diag(l_align))
    # l_unif_log = -jnp.mean(l_unif)

    metrics = {
        "binary_accuracy": jnp.mean((logits > 0) == I),
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "sa_repr_mean": jnp.mean(sa_repr_l2),
        "g_repr_mean": jnp.mean(g_repr_l2),
        "sa_repr_std": jnp.std(sa_repr_l2),
        "g_repr_std": jnp.std(g_repr_l2),
        "logsumexp": logsumexp.mean(),
        "l2_penalty": l2_loss,
        # "l_align": l_align_log,
        # "l_unif": l_unif_log,
    }
    return metrics

def alpha_loss(alpha_params, actor, parametric_action_distribution, training_state, transitions, action_size, key, entropy):
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""

    alpha = jnp.exp(alpha_params["log_alpha"])
    target_entropy = -0.5 * action_size

    alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy)
    return jnp.mean(alpha_loss)

def critic_loss(critic_params, sa_encoder, g_encoder, transitions, state_dim, contrastive_loss_fn_name, energy_fn_name, logsumexp_penalty, l2_penalty, resubs, key):
    sa_encoder_params = critic_params["sa_encoder"]
    g_encoder_params = critic_params["g_encoder"]

    # Compute representations
    sa = jnp.concatenate([transitions.observation[:, :state_dim], transitions.action], axis=-1)
    sa_repr = sa_encoder.apply(sa_encoder_params, sa)
    g = transitions.observation[:, state_dim:]
    g_repr = g_encoder.apply(g_encoder_params, g)

    # InfoNCE
    logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))  # shape = BxB
    loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

    # logsumexp regularisation
    logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
    loss += logsumexp_penalty * jnp.mean(logsumexp ** 2)

    # Compute metrics
    metrics = compute_metrics(logits, sa_repr, g_repr, 0, 0, 0)
    return loss, metrics
    
def actor_loss(actor_params, training_state, actor, sa_encoder, g_encoder, parametric_action_distribution, alpha, transitions, config, state_dim, goal_indices, energy_fn_name, key):
    sample_key, entropy_key, goal_key = jax.random.split(key, 3)
    sa_encoder_params = jax.lax.stop_gradient(training_state.critic_state.params["sa_encoder"])
    g_encoder_params = jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])

    # Compute future state (for goal)
    future_state = transitions.extras["future_state"]
    future_rolled = jnp.roll(future_state, 1, axis=0)
    random_goal_mask = jax.random.bernoulli(goal_key, config.random_goals, shape=(future_state.shape[0], 1))
    future_state = jnp.where(random_goal_mask, future_rolled, future_state)

    # Get state and goal
    state = transitions.observation[:, :state_dim]
    goal = future_state[:, goal_indices]
    sg = jnp.concatenate([state, goal], axis=1)

    # Compute action with policy, given state and goal
    action_mean_and_SD = actor.apply(actor_params, sg)
    x_ts = parametric_action_distribution.sample_no_postprocessing(action_mean_and_SD, sample_key)
    log_prob = parametric_action_distribution.log_prob(action_mean_and_SD, x_ts)
    action = parametric_action_distribution.postprocess(x_ts)

    # Compute representations
    sa = jnp.concatenate([state, action], axis=-1)
    sa_repr = sa_encoder.apply(sa_encoder_params, sa)
    g_repr = g_encoder.apply(g_encoder_params, goal)

    # Compute energy and loss
    q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

    if config.disable_entropy_actor:
        actor_loss = -jnp.mean(q)
    else:
        actor_loss = jnp.mean(alpha * log_prob - (q))

    # Compute metrics
    metrics = {"entropy": -log_prob}
    return jnp.mean(actor_loss), metrics

def actor_step(env, env_state, actor, parametric_action_distribution, actor_params, key, extra_fields=()):
    """
    Executes one step of an actor in the environment by selecting an action based on the
    policy, stepping the environment, and returning the updated state and transition data.

    Parameters
    ----------
    env : Env
        The environment in which the actor operates.
    env_state : State
        The current state of the environment.
    actor : brax.training.types.Policy
        The policy used to select the action.
    parametric_action_distribution : brax.training.distribution.ParametricDistribution
        A tanh normal distribution, used to map the actor's output to an action vector with elements between [-1, 1].
    actor_params : Any
        Parameters for the actor network.
    key : PRNGKey
        A random key for stochastic policy decisions.
    extra_fields : Sequence[str], optional
        A sequence of extra fields to be extracted from the environment state.

    Returns
    -------
    Tuple[State, Transition]
        A tuple containing the new state after taking the action and the transition data
        encompassing observation, action, reward, discount, and extra information.

    """
    action_mean_and_SD = actor.apply(actor_params, env_state.obs)
    action = parametric_action_distribution.sample(action_mean_and_SD, key)
    nstate = env.step(env_state, action)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=env_state.obs,
        action=action,
        reward=nstate.reward,
        discount=1 - nstate.done,
        extras={"policy_extras": {}, "state_extras": state_extras},
    )

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)

def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    policy_lr: float = 1e-4,
    alpha_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    contrastive_loss_fn: str = "binary",
    energy_fn: str = "l2",
    logsumexp_penalty: float = 0.0,
    l2_penalty: float = 0.0,
    resubs: bool = True,
    num_evals: int = 1,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    deterministic_eval: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    unroll_length: int = 50,
    train_step_multiplier: int = 1,
    config: NamedTuple = None,
    use_ln: bool = False,
    h_dim: int = 256,
    n_hidden: int = 2,
    repr_dim: int = 64,
    visualization_interval: int = 5,
):
    """
    Trains a contrastive reinforcement learning agent using the specified environment and parameters.

    This function initializes and manages the training process, including the setup of
    environments, networks, optimizers, replay buffers, and loss functions. It also
    handles the distribution of computation across multiple devices if available and
    configures the training loop to evaluate the agent's performance periodically.

    Parameters:
        environment: envs.Env
            The environment in which the agent will be trained.
        num_timesteps: int
            Total number of timesteps for which the agent will be trained.
        episode_length: int
            Maximum length of an episode.
        action_repeat: int, optional
            Number of times each action is repeated. Default is 1.
        num_envs: int, optional
            Number of parallel environments. Default is 1.
        num_eval_envs: int, optional
            Number of environments for evaluation. Default is 128.
        policy_lr: float, optional
            Learning rate for the policy network. Default is 1e-4.
        alpha_lr: float, optional
            Learning rate for the alpha parameter. Default is 1e-4.
        critic_lr: float, optional
            Learning rate for the critic network. Default is 1e-4.
        seed: int, optional
            Random seed for reproducibility. Default is 0.
        batch_size: int, optional
            Batch size for training. Default is 256.
        contrastive_loss_fn: str, optional
            Type of contrastive loss function. Default is "binary".
        energy_fn: str, optional
            Type of energy function. Default is "l2".
        logsumexp_penalty: float, optional
            Penalty for the log-sum-exp term in the loss function. Default is 0.0.
        l2_penalty: float, optional
            L2 regularization penalty. Default is 0.0.
        resubs: bool, optional
            Whether to use resubstitution in losses.py. Default is True.
        num_evals: int, optional
            Number of evaluation runs. Default is 1.
        min_replay_size: int, optional
            Minimum replay buffer size before starting training. Default is 0.
        max_replay_size: Optional[int], optional
            Maximum replay buffer size. Default is None.
        deterministic_eval: bool, optional
            If True, evaluation is deterministic. Default is False.
        progress_fn: Callable[[int, Metrics], None], optional
            Function to call to report progress. Default is a no-op lambda.
        checkpoint_logdir: Optional[str], optional
            Directory to save checkpoints. Default is None.
        eval_env: Optional[envs.Env], optional
            Evaluation environment. Default is None.
        unroll_length: int, optional
            Length of time to unroll the environment. Default is 50.
        train_step_multiplier: int, optional
            Number of SGD steps multiplier. Default is 1.
        config: NamedTuple, optional
            Configuration settings. Default is None.
        use_ln: bool, optional
            If True, use layer normalization. Default is False.
        h_dim: int, optional
            Dimension of the hidden layers. Default is 256.
        n_hidden: int, optional
            Number of hidden layers. Default is 2.
        repr_dim: int, optional
            Dimension of the representation from the state-action and goal encoders. Default is 64.
        visualization_interval: int, optional
            Number of evals between each visualization of trajectories. Default is 5.


    Raises:
        ValueError
            If the minimum replay size is greater than or equal to the number of timesteps.

    Returns:
        None

    """
    # Reproducibility preparation for (optional) multi-GPU training
    process_id = jax.process_index()
    num_local_devices_to_use = jax.local_device_count()
    device_count = num_local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        num_local_devices_to_use,
        device_count,
    )

    # Sanity checks
    if min_replay_size >= num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")

    if ((episode_length - 1) * num_envs) % batch_size != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs * unroll_length
    num_prefill_actor_steps = min_replay_size // unroll_length + 1
    print("Num_prefill_actor_steps: ", num_prefill_actor_steps)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - min_replay_size >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # Double negative with integer division acts as ceiling function: ceil(num_timesteps - num_prefill_env_steps / (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    wrap_for_training = envs.training.wrap

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    env = TrajectoryIdWrapper(env)
    env = wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)
    unwrapped_env = environment


    obs_size = env.observation_size
    action_size = env.action_size

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_extras = {"state_extras": {"truncation": 0.0, "traj_id": 0.0}, "policy_extras": {}}
    dummy_transition = Transition(observation=dummy_obs, action=dummy_action, reward=0.0, discount=0.0, extras=dummy_extras)
    
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size // device_count,
        num_envs=num_envs,
        episode_length=episode_length,
    )
    replay_buffer = jit_wrap(replay_buffer)
    
    # Network functions
    block_size = 4 # Maybe make this a hyperparameter
    num_blocks = max(1, n_hidden // block_size)
    actor = Net(action_size * 2, h_dim, num_blocks, block_size, use_ln)
    sa_encoder = Net(repr_dim, h_dim, num_blocks, block_size, use_ln)
    g_encoder = Net(repr_dim, h_dim, num_blocks, block_size, use_ln)
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size, ) # Would like to replace this but it's annoying to.

    # Initialize training state (not sure if it makes sense to split and fold local_key here)
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)    
    training_state = _init_training_state(global_key, actor, sa_encoder, g_encoder, env.state_dim, len(env.goal_indices), env.action_size, policy_lr, critic_lr, alpha_lr, num_local_devices_to_use)
    del global_key
    
    # Update functions (may replace later: brax makes it opaque)
    alpha_update = gradient_update_fn(alpha_loss, training_state.alpha_state.tx, pmap_axis_name=_PMAP_AXIS_NAME)
    actor_update = gradient_update_fn(actor_loss, training_state.actor_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    critic_update = gradient_update_fn(critic_loss, training_state.critic_state.tx, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)
    
    def update_step(carry, transitions):
        training_state, key = carry
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha = jnp.exp(training_state.alpha_state.params["log_alpha"])
        (actor_loss, actor_metrics), actor_params, actor_optimizer_state, grad_actor = actor_update(training_state.actor_state.params, training_state, actor, sa_encoder,
                                                                                        g_encoder, parametric_action_distribution, alpha, transitions, 
                                                                                        config, env.state_dim, env.goal_indices, energy_fn, 
                                                                                        key_actor, optimizer_state=training_state.actor_state.opt_state)

        alpha_loss, alpha_params, alpha_optimizer_state, grad_alpha = alpha_update(training_state.alpha_state.params, actor, parametric_action_distribution, training_state, transitions, action_size,
                                                                       key_alpha, actor_metrics['entropy'], optimizer_state=training_state.alpha_state.opt_state,)

        (critic_loss, metrics_crl), critic_params, critic_optimizer_state, grad_critic = critic_update(training_state.critic_state.params, sa_encoder, g_encoder, transitions, env.state_dim,
                                                                                          contrastive_loss_fn, energy_fn, logsumexp_penalty, l2_penalty,
                                                                                          resubs, key_critic, optimizer_state=training_state.critic_state.opt_state)


        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params["log_alpha"]),
            "critic_loss": critic_loss,
            "grad_alpha": jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad_alpha)])),
            "grad_critic": jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad_critic)])),
            "grad_actor": jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad_actor)])),
        }
        metrics.update(metrics_crl)
        metrics.update(actor_metrics)

        new_training_state = TrainingState(
            env_steps=training_state.env_steps,
            gradient_steps=training_state.gradient_steps + 1,
            actor_state=training_state.actor_state.replace(params=actor_params, opt_state=actor_optimizer_state),
            critic_state=training_state.critic_state.replace(params=critic_params, opt_state=critic_optimizer_state),
            alpha_state=training_state.alpha_state.replace(params=alpha_params, opt_state=alpha_optimizer_state),
        )
        return (new_training_state, key), metrics

    def get_experience(actor_params, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(env, env_state, actor, parametric_action_distribution, actor_params, current_key, extra_fields=("truncation", "traj_id"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def training_step(training_state, env_state, buffer_state, key):
        # Collect experience
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)
        env_state, buffer_state = get_experience(training_state.actor_state.params, env_state, buffer_state, experience_key1)
        training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
        
        # Train
        # training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)

        transitions_list = []
        for _ in range(1):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list
        )

        print(
            f"transitions.observation.shape (after {1} episodes per env): {transitions.observation.shape}",
            flush=True)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        vmap_flatten_crl_fn = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))
        transitions = vmap_flatten_crl_fn(config, env, transitions, batch_keys)

        print(f"transitions.observation.shape (after flatten_crl_fn): {transitions.observation.shape}", flush=True)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        print(f"transitions.observation.shape (after first reshape): {transitions.observation.shape}", flush=True)

        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # I added this code, so as to ensure len(transitions.observation) is divisible by batch_size
        num_full_batches = len(transitions.observation) // batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full_batches * batch_size], transitions)
        print(
            f"transitions.observation.shape (after ensuring divisibility by batch_size): {transitions.observation.shape}",
            flush=True)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )

        print(f"transitions.observation.shape (after processing): {transitions.observation.shape}", flush=True)

        if 0 == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key,
                num_total_batches
            )[:800]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices],
                transitions
            )
        print(
            f"transitions.observation.shape (after {0}, selecting {800} batches): {transitions.observation.shape}",
            flush=True)

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)

        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        def f(carry, unused):
            # Collect experience
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state.actor_state.params, env_state, buffer_state, key)
            new_training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (new_training_state, env_state, buffer_state, new_key), ()
        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]
    
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key, update_key = jax.random.split(k, 3)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_training_steps_per_epoch)
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(training_state, env_state, buffer_state, key):
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(training_state, env_state, buffer_state, key)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (training_state, env_state, buffer_state, metrics)

    # Initialization and setup
    ## Env init
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (num_local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    ## Replay buffer init and prefill
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_local_devices_to_use))
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, num_local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_keys)
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    ## Eval init
    global make_policy # Apparently necessary for train() to see make_policy
    make_policy = functools.partial(make_policy, actor, parametric_action_distribution)
    if not eval_env:
        eval_env = environment
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = wrap_for_training(eval_env, episode_length=episode_length, action_repeat=action_repeat)
    evaluator = CrlEvaluator(eval_env, functools.partial(make_policy, deterministic=deterministic_eval), num_eval_envs=num_eval_envs,
                             episode_length=episode_length, action_repeat=action_repeat, key=eval_key)

    ## Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env)

    # Collect/train/eval loop
    current_step = 0
    for eval_epoch_num in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Collect data and train
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, num_local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(training_state, env_state, buffer_state, epoch_keys)
        current_step = int(_unpmap(training_state.env_steps))

        # Logging and evals
        if process_id == 0:
            ## Save policy and critic params
            if checkpoint_logdir:
                params = _unpmap((training_state.actor_state.params, training_state.critic_state.params))
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                brax.io.model.save_params(path, params)

            ## Run evals
            metrics = evaluator.run_evaluation(_unpmap(training_state.actor_state.params), training_metrics)
            logging.info(metrics)
            do_render = (eval_epoch_num % visualization_interval) == 0
            progress_fn(current_step, metrics, make_policy, _unpmap(training_state.actor_state.params), unwrapped_env, do_render)

    # Final validity checks
    ## Verify number of steps is sufficient
    total_steps = current_step
    logging.info("total steps: %s", total_steps)
    assert total_steps >= num_timesteps

    ## If there were no mistakes the training_state should still be identical on all devices
    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
    
    params = _unpmap((training_state.actor_state.params, training_state.critic_state.params))
    return (make_policy, params, metrics)
