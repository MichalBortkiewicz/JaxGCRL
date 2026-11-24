import jax
import jax.numpy as jnp
import numpy as np

from jaxgcrl.agents.common.her import flatten_batch, flatten_batch_with_n_step, n_step_targets
from jaxgcrl.agents.sac.sac import SAC, Transition


class Env:
    state_dim = 1
    goal_indices = jnp.array([0])
    goal_reach_thresh = 0.5


ENV = Env()


def window():
    # one episode of 6 steps ending in a time limit at t=5, then a new episode t=6..7
    # state x_t = t; goal = 3.0 for all; next state = t+1 inside an episode; reset state 100 at the boundary
    T = 8
    x = np.arange(T, dtype=np.float32)
    nx = x + 1
    nx[5] = 100.0  # auto-reset at the time limit
    goal = np.full((T, 1), 3.0, np.float32)
    obs = np.concatenate([x[:, None], goal], 1)
    nobs = np.concatenate([nx[:, None], goal], 1)
    trunc = np.zeros(T, np.float32); trunc[5] = 1
    traj = np.array([0, 0, 0, 0, 0, 0, 1, 1], np.float32)
    reward = (np.abs(nx - 3.0) < 0.5).astype(np.float32)  # env convention: success on the next state
    return Transition(
        observation=jnp.asarray(obs), next_observation=jnp.asarray(nobs), action=jnp.zeros((T, 1)),
        reward=jnp.asarray(reward), discount=jnp.ones(T),
        extras={"state_extras": {"truncation": jnp.asarray(trunc), "traj_id": jnp.asarray(traj)}, "policy_extras": {}},
    )


def test_n1_is_identity():
    cfg = SAC(use_her=False, n_step=1, discounting=0.9)
    tr = window()
    a = flatten_batch(cfg, ENV, tr, jax.random.PRNGKey(0))
    b = flatten_batch_with_n_step(cfg, ENV, tr, jax.random.PRNGKey(0))
    for f in ("observation", "next_observation", "reward", "discount"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))


def test_n3_targets():
    g = 0.9
    cfg = SAC(use_her=False, n_step=3, discounting=g)
    out = n_step_targets(cfg, ENV, flatten_batch(cfg, ENV, window(), jax.random.PRNGKey(0)), g)
    m = np.asarray(out.extras["n_step_used"])
    # t=0..2: 3 steps; t=3: steps 3,4 (5 is the time-limit step) -> 2; t=4: 1 (only step 4); t=5 terminal -> 1
    # t=6: steps 6,7 (window end, not terminal) -> 2; t=7 -> 1
    np.testing.assert_array_equal(m, [3, 3, 3, 2, 1, 1, 2, 1])
    r = np.asarray(out.reward)
    # reward 1 when the next state is 3: transition t=2 (next state 3)
    np.testing.assert_allclose(r[:4], [g**2, g, 1.0, 0.0], rtol=1e-6)
    nx = np.asarray(out.next_observation)[:, 0]
    np.testing.assert_array_equal(nx, [3, 4, 5, 5, 5, 100, 8, 8])
    d = np.asarray(out.discount)
    np.testing.assert_allclose(d, [g**2, g**2, g**2, g, 1, 1, g, 1], rtol=1e-6)
    tr = np.asarray(out.extras["state_extras"]["truncation"])
    np.testing.assert_array_equal(tr, [0, 0, 0, 0, 0, 1, 0, 0])
