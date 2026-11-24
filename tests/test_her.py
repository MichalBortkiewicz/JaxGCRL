"""Tests for jaxgcrl/agents/common/her.py."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.agents.common.her import episode_end_index, flatten_batch
from jaxgcrl.agents.sac.sac import SAC, Transition


class FakeEnv:
    state_dim = 3
    goal_indices = jnp.array([0, 1])
    goal_reach_thresh = 0.5


ENV = FakeEnv()


def make_window(traj, truncation, seed=0):
    """Window of len(traj) steps. State = (10*t, 10*t, t); next state of t = state of t+1 within an episode."""
    T = len(traj)
    rng = np.random.default_rng(seed)
    states = np.stack([10.0 * np.arange(T + 1), 10.0 * np.arange(T + 1), np.arange(T + 1)], 1).astype(np.float32)
    goal = rng.normal(size=(T, 2)).astype(np.float32) * 1000.0  # far away: stored reward 0
    obs = np.concatenate([states[:T], goal], 1)
    next_obs = np.concatenate([states[1:], goal], 1)
    return Transition(
        observation=jnp.asarray(obs),
        next_observation=jnp.asarray(next_obs),
        action=jnp.zeros((T, 2)),
        reward=jnp.zeros((T,)),
        discount=jnp.ones((T,)),
        extras={
            "state_extras": {"truncation": jnp.asarray(truncation, jnp.float32), "traj_id": jnp.asarray(traj, jnp.float32)},
            "policy_extras": {},
        },
    )


TRAJ = [0] * 5 + [1] * 6
TRUNC = [0, 0, 0, 0, 1] + [0] * 6


def test_episode_end_index():
    np.testing.assert_array_equal(np.asarray(episode_end_index(jnp.asarray(TRAJ))), [4] * 5 + [10] * 6)


def test_future_goals_from_same_episode_later_step():
    cfg = SAC(use_her=True, her_strategy="future", her_ratio=1.0)
    tr = make_window(TRAJ, TRUNC)
    for s in range(50):
        out = flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(s))
        goals = np.asarray(out.observation[:, 3:])
        relab = np.asarray(out.extras["her_relabeled"])
        ends = [4] * 5 + [10] * 6
        for t in range(len(TRAJ)):
            if t in (4, 10):  # last step of each episode: keeps its goal
                assert relab[t] == 0
                np.testing.assert_allclose(goals[t], np.asarray(tr.observation[t, 3:]))
            else:
                assert relab[t] == 1
                tp = goals[t][0] / 10.0
                assert abs(tp - round(tp)) < 1e-6
                assert t < round(tp) <= ends[t]
        # next_observation carries the same goal
        np.testing.assert_allclose(np.asarray(out.next_observation[:, 3:]), goals)


def test_her_ratio_frequency():
    cfg = SAC(use_her=True, her_strategy="future", her_ratio=0.8)
    tr = make_window(TRAJ, TRUNC)
    fn = jax.jit(jax.vmap(functools.partial(flatten_batch, cfg, ENV, tr)))
    out = fn(jax.random.split(jax.random.PRNGKey(0), 2000))
    relab = np.asarray(out.extras["her_relabeled"])
    eligible = np.array([t not in (4, 10) for t in range(len(TRAJ))])
    frac = relab[:, eligible].mean()
    assert abs(frac - 0.8) < 0.01, frac
    assert relab[:, ~eligible].sum() == 0


def test_next_step_goal_gives_reward_one():
    # episode of length 2 inside the window: t=0's only candidate is t'=1, whose state equals next_obs[0]
    traj = [0, 0, 1, 1, 1]
    cfg = SAC(use_her=True, her_strategy="future", her_ratio=1.0)
    tr = make_window(traj, [0, 1, 0, 0, 0])
    out = flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(0))
    assert float(out.extras["her_relabeled"][0]) == 1.0
    assert float(out.reward[0]) == 1.0
    # "current" convention: state at t=0 is 10 units away from state at t=1 -> reward 0
    cfg_cur = SAC(use_her=True, her_strategy="future", her_ratio=1.0, her_reward_obs="current")
    assert float(flatten_batch(cfg_cur, ENV, tr, jax.random.PRNGKey(0)).reward[0]) == 0.0


def test_non_relabelled_unchanged():
    cfg = SAC(use_her=True, her_strategy="future", her_ratio=0.5)
    tr = make_window(TRAJ, TRUNC)
    tr = tr._replace(reward=jnp.arange(len(TRAJ), dtype=jnp.float32) + 7.0)  # recognisable stored rewards
    out = flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(3))
    keep = np.asarray(out.extras["her_relabeled"]) == 0
    assert keep.any()
    np.testing.assert_allclose(np.asarray(out.observation)[keep], np.asarray(tr.observation)[keep])
    np.testing.assert_allclose(np.asarray(out.next_observation)[keep], np.asarray(tr.next_observation)[keep])
    np.testing.assert_allclose(np.asarray(out.reward)[keep], np.asarray(tr.reward)[keep])
    np.testing.assert_allclose(np.asarray(out.discount), np.asarray(tr.discount))


# Verbatim copy of the original JaxGCRL function (sac.py:89-139 at a2740ce), for the regression test.
@functools.partial(jax.jit, static_argnames=["config", "env"])
def original_flatten_batch(config, env, transition, sample_key):
    if config.use_her:
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )
        final_step_mask = (
            is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        )
        final_step_mask = jnp.logical_and(
            final_step_mask,
            transition.extras["state_extras"]["truncation"][None, :],
        )
        non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]
        new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
        binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)
        new_goals = (
            binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
            + jnp.logical_not(binary_mask)[:, None]
            * transition.observation[new_goals_idx][:, env.state_dim :]
        )
        state = transition.observation[:, : env.state_dim]
        new_obs = jnp.concatenate([state, new_goals], axis=1)
        dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
        new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)
        next_state = transition.next_observation[:, : env.state_dim]
        new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)
        return transition._replace(
            observation=jnp.squeeze(new_obs),
            next_observation=jnp.squeeze(new_next_obs),
            reward=jnp.squeeze(new_reward),
        )
    return transition


@pytest.mark.parametrize("use_her", [False, True])
def test_final_matches_original(use_her):
    cfg = SAC(use_her=use_her)  # her_strategy defaults to "final"
    tr = make_window(TRAJ, TRUNC)
    new = flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(0))
    old = original_flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(0))
    for f in ("observation", "next_observation", "reward", "action", "discount"):
        np.testing.assert_array_equal(np.asarray(getattr(new, f)), np.asarray(getattr(old, f)))
