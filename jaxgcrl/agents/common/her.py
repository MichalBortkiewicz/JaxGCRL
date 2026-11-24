"""Hindsight relabelling shared by SAC and TD3.

`flatten_batch` is called once per sampled window (vmapped over windows), exactly as the original
per-agent functions were (sac.py:89-139 and td3.py:92-141 at JaxGCRL a2740ce).

Strategies (agent config field `her_strategy`, default "final"):
  "final"  - the original JaxGCRL code, unchanged: relabel to the state at the first later time-limit
             step of the same trajectory inside the window; reward recomputed from the *current*
             observation (sac.py:126-127).
  "future" - standard HER "future" (Andrychowicz et al. 2017): for transition t, draw t' uniformly
             from the later steps of the same episode inside the window; with probability `her_ratio`
             replace the goal by observation[t'][goal_indices]. If t has no later step in its
             episode inside the window, keep the original goal. Relabelled rewards follow
             `her_reward_obs` ("next": the env's own convention, success measured on the post-step
             observation; "current": the old JaxGCRL convention). Non-relabelled transitions keep
             their stored reward. `discount` and `extras` are unchanged.

Both strategies add extras["her_relabeled"] (float 0/1 per transition) for diagnostics.
"""

import functools

import jax
import jax.numpy as jnp


def _final_strategy(env, transition):
    """Original JaxGCRL final-state HER (sac.py:91-137 at a2740ce), unchanged apart from returning the mask."""
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )

    # final_step_mask.shape == (seq_len, seq_len)
    final_step_mask = (
        is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    )
    final_step_mask = jnp.logical_and(
        final_step_mask,
        transition.extras["state_extras"]["truncation"][None, :],
    )
    non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

    # If final state is not present use original goal (i.e. don't change anything)
    new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
    binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

    new_goals = (
        binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
        + jnp.logical_not(binary_mask)[:, None] * transition.observation[new_goals_idx][:, env.state_dim :]
    )

    # Transform observation
    state = transition.observation[:, : env.state_dim]
    new_obs = jnp.concatenate([state, new_goals], axis=1)

    # Recalculate reward
    dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
    new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)

    # Transform next observation
    next_state = transition.next_observation[:, : env.state_dim]
    new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

    extras = dict(transition.extras)
    extras["her_relabeled"] = binary_mask.astype(jnp.float32)
    return transition._replace(
        observation=jnp.squeeze(new_obs),
        next_observation=jnp.squeeze(new_next_obs),
        reward=jnp.squeeze(new_reward),
        extras=extras,
    )


def episode_end_index(traj_id: jnp.ndarray) -> jnp.ndarray:
    """For each t, the last index j >= t in the window such that traj_id[t..j] are all equal to traj_id[t]."""
    seq_len = traj_id.shape[0]

    def body(carry, x):
        next_end, next_tid = carry
        t, tid = x
        end = jnp.where(tid == next_tid, next_end, t)
        return (end, tid), end

    init = (jnp.asarray(seq_len - 1, dtype=jnp.int32), traj_id[-1])
    _, ends = jax.lax.scan(body, init, (jnp.arange(seq_len, dtype=jnp.int32), traj_id), reverse=True)
    return ends


def sample_future_indices(traj_id: jnp.ndarray, key: jax.Array):
    """Uniform t' in {t+1, ..., end_of_episode(t)}; has_candidate is False when there is no such t'."""
    seq_len = traj_id.shape[0]
    t = jnp.arange(seq_len, dtype=jnp.int32)
    ends = episode_end_index(traj_id)
    n_cand = ends - t
    has_candidate = n_cand > 0
    u = jax.random.uniform(key, (seq_len,))
    offset = 1 + jnp.floor(u * jnp.maximum(n_cand, 1)).astype(jnp.int32)
    offset = jnp.clip(offset, 1, jnp.maximum(n_cand, 1))
    t_prime = jnp.where(has_candidate, t + offset, t)
    return t_prime, has_candidate


def _future_strategy(config, env, transition, sample_key):
    key_idx, key_ratio = jax.random.split(sample_key)
    traj_id = transition.extras["state_extras"]["traj_id"]
    t_prime, has_candidate = sample_future_indices(traj_id, key_idx)
    relabel = jnp.logical_and(has_candidate, jax.random.uniform(key_ratio, has_candidate.shape) < config.her_ratio)

    old_goal = transition.observation[:, env.state_dim :]
    candidate_goal = transition.observation[t_prime][:, env.goal_indices]
    goal = jnp.where(relabel[:, None], candidate_goal, old_goal)

    state = transition.observation[:, : env.state_dim]
    next_state = transition.next_observation[:, : env.state_dim]
    new_obs = jnp.concatenate([state, goal], axis=1)
    new_next_obs = jnp.concatenate([next_state, goal], axis=1)

    reward_source = transition.next_observation if config.her_reward_obs == "next" else transition.observation
    dist = jnp.linalg.norm(reward_source[:, env.goal_indices] - goal, axis=1)
    relabeled_reward = jnp.asarray(dist < env.goal_reach_thresh, dtype=transition.reward.dtype)
    reward = jnp.where(relabel, relabeled_reward, transition.reward)

    extras = dict(transition.extras)
    extras["her_relabeled"] = relabel.astype(jnp.float32)
    return transition._replace(observation=new_obs, next_observation=new_next_obs, reward=reward, extras=extras)


def n_step_targets(config, env, transition, discounting):
    """Turn each transition t of a window into an m-step transition, m = min(n, steps left in its episode).

    Uses the (relabelled) goal g_t of transition t for every step: r_{t,k} = 1[||s'_{t+k}[goal] - g_t|| < thr]
    (the env's next-state convention, Fact B), k = 0..m-1, with r_{t,0} the transition's own reward.
    The window stops *before* the episode's terminal step (fall or time limit), whose stored
    next_observation is the reset state of the next episode; a terminal step t itself keeps m = 1
    (its own reward, discount and truncation flag). Output, as Brax's SAC/TD3 losses expect:
        reward = sum_k gamma^k r_{t,k},  next_observation = s'_{t+m-1} with goal g_t,
        discount = gamma^(m-1) * discount_{t+m-1},  truncation = truncation_{t+m-1}
    so that reward + discount * gamma * V(next_observation) is the m-step target. Entropy bonuses of the
    intermediate steps are not added (as in LAC, Kang et al. 2026).
    """
    n = config.n_step
    T = transition.observation.shape[0]
    t = jnp.arange(T)
    traj = transition.extras["state_extras"]["traj_id"]
    trunc = transition.extras["state_extras"]["truncation"]
    terminal = jnp.logical_or(transition.discount == 0, trunc > 0)
    ends = episode_end_index(traj)
    end_is_terminal = terminal[ends]
    avail = ends - t + jnp.where(end_is_terminal, 0, 1)  # usable steps starting at t (excluding terminal)
    m = jnp.clip(jnp.minimum(n, avail), 1, n)

    goal = transition.observation[:, env.state_dim :]
    rewards = [transition.reward]
    for k in range(1, n):
        idx = jnp.minimum(t + k, T - 1)
        dist = jnp.linalg.norm(transition.next_observation[idx][:, env.goal_indices] - goal, axis=1)
        rewards.append(jnp.asarray(dist < env.goal_reach_thresh, dtype=transition.reward.dtype))
    rewards = jnp.stack(rewards, 1)  # (T, n)
    k = jnp.arange(n)[None, :]
    weights = jnp.where(k < m[:, None], discounting**k, 0.0)
    reward = jnp.sum(rewards * weights, axis=1)

    last = jnp.minimum(t + m - 1, T - 1)
    next_state = transition.next_observation[last][:, : env.state_dim]
    next_obs = jnp.concatenate([next_state, goal], axis=1)
    discount = (discounting ** (m - 1)) * transition.discount[last]
    extras = dict(transition.extras)
    extras["state_extras"] = dict(extras["state_extras"], truncation=trunc[last])
    extras["n_step_used"] = m.astype(jnp.float32)
    return transition._replace(reward=reward, next_observation=next_obs, discount=discount, extras=extras)


def flatten_batch_with_n_step(config, env, transition, sample_key):
    out = flatten_batch(config, env, transition, sample_key)
    if getattr(config, "n_step", 1) > 1:
        out = n_step_targets(config, env, out, config.discounting)
    return out


@functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_batch(config, env, transition, sample_key):
    if not config.use_her:
        extras = dict(transition.extras)
        extras["her_relabeled"] = jnp.zeros_like(transition.reward, dtype=jnp.float32)
        return transition._replace(extras=extras)
    strategy = getattr(config, "her_strategy", "final")
    if strategy == "final":
        return _final_strategy(env, transition)
    if strategy == "future":
        return _future_strategy(config, env, transition, sample_key)
    raise ValueError(f"Unknown her_strategy {strategy!r}")
