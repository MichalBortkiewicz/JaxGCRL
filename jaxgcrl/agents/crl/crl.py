import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.visualize import visualize_goals_2d, visualize_kde_heatmap, visualize_q_function_2d

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder
from .goals import GoalProposer, ReplayBufferGoalProposal, MediumEnergyGoalProposal, MetricPreservationGoalProposal, FisherTraceGoalProposal, QEpistemicGoalProposal, MEGAGoalProposal, OMEGAGoalProposal, UCGRGoalProposal,DISCOVERGoalProposal, mix_goals

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner"""
    optimal_goal_proposal_prob: jnp.ndarray
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    # transition.observations has size (episode_length, obs_dim)
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len wheree a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index
    proposed_goals = transition.observation[:, -len(goal_indices):]

    def last_state_for_each_step(obs, traj_ids):
        seq_len = obs.shape[0]
        def last_state_for_t(i):
            mask = traj_ids == traj_ids[i]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs[last_idx]
        return jax.vmap(last_state_for_t)(jnp.arange(seq_len))
    
    def get_intermediate_trajectory_states(obs, traj_ids):
        """Returns states at 1/3 and 2/3 of remaining trajectory for each timestep"""
        seq_len = obs.shape[0]
        obs_dim = obs.shape[1]
        
        def intermediate_states_for_t(i, num_intermediate):
            # Mask for same trajectory AND future timesteps (including current)
            same_traj_mask = traj_ids == traj_ids[i]
            future_mask = jnp.arange(seq_len) >= i
            mask = same_traj_mask & future_mask

            # Get sorted valid indices for future steps
            indices = jnp.where(mask, jnp.arange(seq_len), seq_len)
            sorted_indices = jnp.sort(indices)
            num_future = jnp.sum(mask)

            # Compute evenly spaced fractional positions in (0, 1)
            # e.g. for 2 → [1/3, 2/3], for 3 → [1/4, 1/2, 3/4]
            fractions = (jnp.arange(1, num_intermediate + 1) / (num_intermediate + 1))

            # Map fractions to integer positions within the valid range
            idxs = jnp.floor(fractions * num_future).astype(jnp.int32)
            idxs = jnp.clip(idxs, 0, jnp.maximum(num_future - 1, 0))

            # Gather actual indices in the trajectory
            actual_idxs = sorted_indices[idxs]

            # Get the corresponding future states (with padding for no valid futures)
            def get_state(idx):
                return jnp.where(num_future > 0, obs[idx], jnp.zeros(obs_dim))

            states = jax.vmap(get_state)(actual_idxs)
            return states
                
        return jax.vmap(functools.partial(intermediate_states_for_t, num_intermediate=6))(jnp.arange(seq_len))

    traj_ids = transition.extras["state_extras"]["traj_id"]  # shape (seq_len,)
    last_traj_state = last_state_for_each_step(transition.observation, traj_ids)
    intermediate_traj = get_intermediate_trajectory_states(transition.observation, traj_ids)

    def is_last_occurrence(i):
        traj_id = traj_ids[i]
        future_mask = jnp.arange(seq_len) > i
        has_future_same_id = jnp.any((traj_ids == traj_id) & future_mask)
        return ~has_future_same_id
    
    last_traj_state_mask = jax.vmap(is_last_occurrence)(jnp.arange(seq_len))

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            "was_proposed_goal_mask": jnp.squeeze(transition.extras["state_extras"]["was_proposed_goal_mask"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "proposed_goals": proposed_goals[:-1],
        "last_traj_state": last_traj_state[:-1],
        "intermediate_traj": intermediate_traj[:-1],
        "last_traj_state_mask": last_traj_state_mask[:-1],
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRL:
    """Contrastive Reinforcement Learning (CRL) agent."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Proportion of proposed goals coming from the goal proposal algorithm
    goal_proposal_prob: float = 0.0
    # If fraction of goals from the replay buffer should be computed adaptiveally; note that this causes goal_proposal_prob to be ignored
    use_adaptive_mixing: bool = False
    # Adaptive mixing momentum term
    adaptive_mixing_momentum: float = 0.0
    # Number of env steps to wait before starting adaptive mixing
    adaptive_mixing_warmup_steps: int = 0 
    # Number of env steps to wait before proposing goals from the goal proposal algorithm
    goal_proposal_warmup_steps: int = 0
    # Whether we should interpolate to 100% environment goals during training
    interpolate_to_env_goals: bool = False
    # What goal selection percentile to use for MediumEnergyGoalProposal
    goal_selection_percentile: float = 0.5
    # Which goal proposer to use
    goal_proposer_name: Literal["quantile", "replay_buffer", "metric", "metric_one_env_goal", "waypoint_ratio", "waypoint_ratio_one_env_goal", "max_waypoint_ratio", "fisher_trace", "fisher_trace_actor", "fisher_trace_combined", "q_epistemic", "mega", "omega", "ucgr", "discover"] = "replay_buffer"
    # For metric proposal whether to use KDE correction term
    use_kde_correction: bool = False
    # Whether to zero out the goals in metric proposal
    zero_out_cand_goals: bool = True
    # Whether to zero out the current state when computing energy terms
    zero_out_state: bool = False
    # Whether to propose environment goals instead of waypoint goals (for max_waypoint_ratio)
    propose_env_goals: bool = False
    # Number of critics in the ensemble
    use_critic_ensemble: bool = False
    num_critic_ensemble: int = 1
    # Whether to use environment goals (True) or replay buffer final states (False) for q_epistemic
    q_epistemic_use_env_goals: bool = False
    # Whether to zero-center each critic's predictions before computing std (removes translational offset)
    q_zero_center: bool = False
    # Temperature for softmax goal sampling over M matrix (0 = greedy, >0 = softmax sampling)
    goal_sampling_temperature: float = 1.0
    # Target rollout probability for DISCOVER
    discover_target_prob: float = 0.5

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        logging.info("Num env: %d", config.num_envs)

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)
        
        # Initialize proposed_goals tracking in env_state.info
        # These will be updated only at episode boundaries
        env_state.info["proposed_goals"] = env_state.obs[:, -len(train_env.goal_indices):]
        env_state.info["was_proposed_goal_mask"] = jnp.zeros((config.num_envs,))
        # Track traj_id to detect episode boundaries (set to -1 so first step triggers new episode)
        env_state.info["last_traj_id"] = env_state.info["traj_id"] - 1

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
       
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        
        # Initialize critic params - use ensemble if q_epistemic, otherwise single critic
        if self.use_critic_ensemble:
            # Initialize ensemble of critics with different random keys
            sa_keys = jax.random.split(sa_key, self.num_critic_ensemble)
            g_keys = jax.random.split(g_key, self.num_critic_ensemble)
            sa_encoder_params = [sa_encoder.init(k, np.ones([1, state_size + action_size])) for k in sa_keys]
            g_encoder_params = [g_encoder.init(k, np.ones([1, goal_size])) for k in g_keys]
        else:
            # Single critic
            sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size]))
            g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            optimal_goal_proposal_prob=jnp.array(self.goal_proposal_prob),
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                    "was_proposed_goal_mask": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        if self.goal_proposer_name == "quantile":
            goal_proposer = MediumEnergyGoalProposal(
                energy_fn_name=self.energy_fn,
                selection_percentile=self.goal_selection_percentile
            )
        elif self.goal_proposer_name == "replay_buffer":
            goal_proposer = ReplayBufferGoalProposal()
        elif self.goal_proposer_name == "metric":
            goal_proposer = MetricPreservationGoalProposal(energy_fn_name=self.energy_fn, use_waypoint_difficulty=True, use_one_env_goal=False, use_kde_correction=self.use_kde_correction, zero_out_cand_goals=self.zero_out_cand_goals, zero_out_state=self.zero_out_state, goal_sampling_temperature=self.goal_sampling_temperature)
        elif self.goal_proposer_name == "metric_one_env_goal":
            goal_proposer = MetricPreservationGoalProposal(energy_fn_name=self.energy_fn,use_waypoint_difficulty=True, use_one_env_goal=True, use_kde_correction=self.use_kde_correction, zero_out_cand_goals=self.zero_out_cand_goals, zero_out_state=self.zero_out_state, goal_sampling_temperature=self.goal_sampling_temperature)
        elif self.goal_proposer_name == "waypoint_ratio":
            goal_proposer = MetricPreservationGoalProposal(energy_fn_name=self.energy_fn, use_waypoint_difficulty=False, use_one_env_goal=False, use_kde_correction=False, zero_out_cand_goals=self.zero_out_cand_goals, zero_out_state=self.zero_out_state, goal_sampling_temperature=self.goal_sampling_temperature)
        elif self.goal_proposer_name == "waypoint_ratio_one_env_goal":
            goal_proposer = MetricPreservationGoalProposal(energy_fn_name=self.energy_fn, use_waypoint_difficulty=False, use_one_env_goal=True, use_kde_correction=False, zero_out_cand_goals=self.zero_out_cand_goals, zero_out_state=self.zero_out_state, goal_sampling_temperature=self.goal_sampling_temperature)
        elif self.goal_proposer_name == "max_waypoint_ratio":
            goal_proposer = MetricPreservationGoalProposal(energy_fn_name=self.energy_fn, use_waypoint_difficulty=False, use_one_env_goal=False, use_max=True, use_kde_correction=False, zero_out_cand_goals=self.zero_out_cand_goals, zero_out_state=self.zero_out_state, propose_env_goals=self.propose_env_goals, goal_sampling_temperature=self.goal_sampling_temperature)
        elif self.goal_proposer_name == "fisher_trace":
            goal_proposer = FisherTraceGoalProposal(energy_fn_name=self.energy_fn, use_critic_gradients=True, use_actor_gradients=False, temperature=self.goal_sampling_temperature, propose_env_goals=self.propose_env_goals)
        elif self.goal_proposer_name == "fisher_trace_actor":
            goal_proposer = FisherTraceGoalProposal(energy_fn_name=self.energy_fn, use_critic_gradients=False, use_actor_gradients=True, temperature=self.goal_sampling_temperature, propose_env_goals=self.propose_env_goals)
        elif self.goal_proposer_name == "fisher_trace_combined":
            goal_proposer = FisherTraceGoalProposal(energy_fn_name=self.energy_fn, use_critic_gradients=True, use_actor_gradients=True, temperature=self.goal_sampling_temperature, propose_env_goals=self.propose_env_goals)
        elif self.goal_proposer_name == "q_epistemic":
            goal_proposer = QEpistemicGoalProposal(
                energy_fn_name=self.energy_fn,
                num_ensemble=self.num_critic_ensemble,
                use_env_goals=self.q_epistemic_use_env_goals,
                zero_center=self.q_zero_center
            )
        elif self.goal_proposer_name == "mega":
            goal_proposer = MEGAGoalProposal(
                energy_fn_name=self.energy_fn,
            )
        elif self.goal_proposer_name == "omega":
            goal_proposer = OMEGAGoalProposal(
                energy_fn_name=self.energy_fn,
            )
        elif self.goal_proposer_name == "ucgr":
            goal_proposer = UCGRGoalProposal(
                energy_fn_name=self.energy_fn,
                num_samples=100
            )
        elif self.goal_proposer_name == "discover":
            goal_proposer = DISCOVERGoalProposal(
                energy_fn_name=self.energy_fn,
                num_ensemble=self.num_critic_ensemble,
                alpha_0=0.5,
                target_prob=self.discover_target_prob,
            )
        else:
            raise ValueError(f"Unknown goal proposer: {self.goal_proposer_name}")

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, env, env_state, proposed_goals, was_proposed_goal_mask, key, extra_fields):
            new_obs = env_state.obs.at[:, -len(env.goal_indices):].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)

            means, log_stds = actor.apply(actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras["was_proposed_goal_mask"] = was_proposed_goal_mask

            # nstate.obs has shape (batch_size, obs_dim)
            return nstate, Transition(
                observation=new_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def propose_goals_for_new_episodes(env_state, training_state, buffer_state, key):
            """Propose new goals only for environments that just started a new episode."""
            proposal_key, mix_key = jax.random.split(key)
            
            # Compare current traj_id with stored traj_id to detect resets
            current_traj_id = env_state.info["traj_id"]
            stored_traj_id = env_state.info.get("last_traj_id", current_traj_id - 1)
            is_new_episode = current_traj_id != stored_traj_id  # shape (num_envs,)
            
            # Propose new goals
            new_goals, buffer_state = goal_proposer.propose_goals(
                replay_buffer, buffer_state,
                training_state, train_env, env_state,
                proposal_key,
                actor, training_state.actor_state.params, training_state.critic_state.params,
                sa_encoder, g_encoder
            )
            original_goals = env_state.obs[:, -len(train_env.goal_indices):]

            # Compute mixing probability
            if self.use_adaptive_mixing:
                curr_goal_proposal_prob = jax.lax.cond(
                    training_state.env_steps >= self.adaptive_mixing_warmup_steps,
                    lambda: training_state.optimal_goal_proposal_prob,
                    lambda: 0.5,
                )
            elif self.interpolate_to_env_goals:
                progress_frac = training_state.env_steps / config.total_env_steps
                curr_goal_proposal_prob = self.goal_proposal_prob * (1 - progress_frac)
            else:
                curr_goal_proposal_prob = self.goal_proposal_prob

            # Mix goals for new episodes
            mixed_goals, use_proposed_mask = mix_goals(original_goals, new_goals, curr_goal_proposal_prob, mix_key)

            # Apply warmup: only use proposed goals after warmup period
            should_use_proposed = training_state.env_steps >= self.goal_proposal_warmup_steps
            new_proposed_goals = jax.lax.cond(
                should_use_proposed,
                lambda: mixed_goals,
                lambda: original_goals,
            )
            new_was_proposed_mask = jax.lax.cond(
                should_use_proposed,
                lambda: use_proposed_mask.squeeze(-1),
                lambda: jnp.zeros_like(use_proposed_mask.squeeze(-1)),
            )

            # Only update goals for environments that are starting a new episode
            # Keep existing goals for environments mid-episode
            proposed_goals = jnp.where(
                is_new_episode[:, None],
                new_proposed_goals,
                env_state.info["proposed_goals"]
            )
            was_proposed_goal_mask = jnp.where(
                is_new_episode,
                new_was_proposed_mask,
                env_state.info["was_proposed_goal_mask"]
            )

            return proposed_goals, was_proposed_goal_mask, buffer_state

        @jax.jit
        def get_experience(actor_state, env_state, training_state, buffer_state, key):        
            @jax.jit
            def f(carry, unused_t):
                env_state, training_state, buffer_state, current_key = carry
                current_key, next_key, proposal_key = jax.random.split(current_key, 3)
                
                # Check for new episodes and propose goals only for those environments
                proposed_goals, was_proposed_goal_mask, buffer_state = propose_goals_for_new_episodes(
                    env_state, training_state, buffer_state, proposal_key
                )
                
                # Store the current traj_id before the step (to detect changes after step)
                pre_step_traj_id = env_state.info["traj_id"]
                
                # Update env_state.info with the current goals
                env_state.info["proposed_goals"] = proposed_goals
                env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
                
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    proposed_goals,
                    was_proposed_goal_mask,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                
                # Preserve info in the new state returned by env.step()
                # last_traj_id is set to pre-step value so we can detect if traj_id changed
                env_state.info["proposed_goals"] = proposed_goals
                env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
                env_state.info["last_traj_id"] = pre_step_traj_id
                
                return (env_state, training_state, buffer_state, next_key), transition

            # data.observation has shape (unroll_length, batch_size, obs_size)
            (env_state, _, buffer_state, _), data = jax.lax.scan(
                f, (env_state, training_state, buffer_state, key), (), length=self.unroll_length
            )

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    env_state,
                    training_state,
                    buffer_state,
                    key,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # update buffer
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                env_state,
                training_state,
                buffer_state,
                experience_key1,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            # transitions.observation has shape (num_envs, episode_length, obs_dim)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                transitions,
                batch_keys,
            )
            # transitions.observation has shape (num_envs, episode_length, obs_dim)

            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )
            # Shape of obs is (num_envs * episode_length, obs_dim) after flattening

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )
            # Shape of transitions.observation is (..., batch_size, obs_dim)
            
            last_batch = transitions

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)

            # Adaptive mixing estimation - only if enabled
            if self.use_adaptive_mixing:
                # Get the last batch metrics which contain gradient info
                last_metrics = jax.tree_map(lambda x: x[-1], metrics)
                
                S1 = last_metrics.get('rb_grad_trvar') # tr Var d_rb
                S2 = last_metrics.get('env_grad_trvar')  # tr Var d_env
                D = last_metrics.get('env_rb_bias_squared') # norm(E[d_env - d_rb])^2
                B = self.batch_size
                
                # Compute optimal alpha
                numerator = S1 - S2 + D
                denominator = 2 * D
                mixing_star = numerator / (denominator + 1e-8)
                mixing_star = jnp.clip(mixing_star, 0.0, 1.0)

                smoothed_mixing = (
                    self.adaptive_mixing_momentum * training_state.optimal_goal_proposal_prob 
                    + (1 - self.adaptive_mixing_momentum) * mixing_star
                )

                training_state = training_state.replace(optimal_goal_proposal_prob=smoothed_mixing)

                metrics['adaptive_mixing/mixing_raw'] = mixing_star
                metrics['adaptive_mixing/mixing_smoothed'] = smoothed_mixing

            return (
                training_state,
                env_state,
                buffer_state,
                last_batch
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k, _ = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                        last_batch
                    ),
                    metrics,
                ) = training_step(ts, es, bs, train_key)
                # Keep last_batch in carry to avoid stacking all batches in memory
                return (ts, es, bs, k, last_batch), metrics

            # Run one step to get initial last_batch structure for carry
            key, first_key = jax.random.split(key)
            ((training_state, env_state, buffer_state, init_batch), first_metrics) = training_step(
                training_state, env_state, buffer_state, first_key
            )

            (training_state, env_state, buffer_state, _, last_batch), rest_metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, init_batch),
                (),
                length=num_training_steps_per_epoch - 1,
            )

            # Combine metrics from first step with rest
            metrics = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a[None], b]),
                first_metrics,
                rest_metrics,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics, last_batch
        
        def visualize_goals(train_env, transitions, actor_state, critic_state, sa_encoder, g_encoder, energy_fn, num_samples, wandb_key):
            # Shape is now (episode_len-1, batch_size, ...) since we only keep the last training step's batch
            obs = transitions.observation # (episode_len-1, batch_size, obs_dim)
            future_state = transitions.extras["future_state"] # (episode_len-1, batch_size, obs_dim)
            last_traj_state = transitions.extras["last_traj_state"][:, :, :state_size] # (episode_len-1, batch_size, state_size)
            last_traj_state_flat = last_traj_state.reshape(-1, state_size)
            intermediate_traj = transitions.extras["intermediate_traj"] # (episode_len-1, batch_size, num_intermediate_states, obs_dim)
            
            states = obs[:, :, :state_size].reshape(-1, state_size)
            contrastive_goals = future_state[:, :, train_env.goal_indices].reshape(-1, len(train_env.goal_indices))
            proposed_goals = transitions.extras["proposed_goals"].reshape(-1, len(train_env.goal_indices))
            
            # Flatten intermediate trajectories to shape (total_samples, num_intermediate_states, obs_dim)
            intermediate_traj_flat = intermediate_traj.reshape(-1, intermediate_traj.shape[-2], intermediate_traj.shape[-1])
            
            total_samples = states.shape[0]
            if num_samples > total_samples:
                num_samples = total_samples
            
            sample_indices = np.random.choice(total_samples, num_samples, replace=False)
            
            start_xy = states[sample_indices][:, train_env.goal_indices]
            cont_xy = contrastive_goals[sample_indices][:, train_env.goal_indices]
            prop_xy = proposed_goals[sample_indices]
            lts_xy = last_traj_state_flat[sample_indices][:, train_env.goal_indices]
            intermediate_xy = intermediate_traj_flat[sample_indices][:, :, train_env.goal_indices]
            last_traj_state_mask = transitions.extras["last_traj_state_mask"].reshape(-1)

            visualize_goals_2d(start_xy, cont_xy, prop_xy, lts_xy, intermediate_xy, f"{wandb_key}/state_goal_plot", x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds)

            visualize_kde_heatmap(proposed_goals[last_traj_state_mask], "Proposed Goals", f"{wandb_key}/proposed_goal_heatmap", x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds)

            visualize_kde_heatmap(last_traj_state_flat[last_traj_state_mask][:, train_env.goal_indices], "Final States", f"{wandb_key}/final_states_heatmap", x_bounds=train_env.x_bounds, y_bounds=train_env.y_bounds)

            for i in range(min(3, num_samples)):  # Visualize Q-function for up to 3 samples
                idx = sample_indices[i]
                visualize_q_function_2d(
                    actor, sa_encoder, g_encoder, actor_state.params, critic_state.params,
                    states[idx],
                    train_env.goal_indices,
                    train_env.x_bounds, train_env.y_bounds,
                    f"{wandb_key}/q_function_sample_{i}",
                    energy_fn
                )

            logging.info(f"Plotted visualizations at env step {training_state.env_steps.item()}")
            
        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics, last_batch = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            visualize_goals(train_env, last_batch, training_state.actor_state, training_state.critic_state, sa_encoder, g_encoder, self.energy_fn, num_samples=5, wandb_key="training")

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
            else:
                params = None

        total_steps = current_step
        # assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
