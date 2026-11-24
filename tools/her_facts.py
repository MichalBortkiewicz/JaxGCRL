"""Facts A, B, C of 02_IMPLEMENTATION_SPEC.md Task 5.1, measured on real environments with random actions.

A: at a time-limit step, is next_observation the first observation of the next episode (auto-reset)?
B: is the env reward success measured on next_observation or on the current observation?
   (The goal threshold is temporarily raised to the median goal distance so that ~half the steps are
   successes; otherwise random actions give almost no positive rewards and the test is uninformative.)
C: with the original final-state HER, how many relabelled transitions get reward 1, and how many of those
   are time-limit steps (which Brax's SAC loss masks out)? Same counts for future HER.
Usage: python tools/her_facts.py --envs ant_big_maze humanoid arm_push_easy --out reports/facts_her.json
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

from jaxgcrl.agents.common.her import flatten_batch
from jaxgcrl.agents.sac.sac import SAC, Transition
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.env import create_env


def rollout(raw_env, num_envs, steps, seed, episode_length=1001):
    env = envs.training.wrap(TrajectoryIdWrapper(raw_env), episode_length=episode_length, action_repeat=1)
    key = jax.random.PRNGKey(seed)
    state = jax.jit(env.reset)(jax.random.split(key, num_envs))

    def step(carry, _):
        s, k = carry
        k, ka = jax.random.split(k)
        a = jax.random.uniform(ka, (num_envs, env.action_size), minval=-1.0, maxval=1.0)
        ns = env.step(s, a)
        tr = dict(obs=s.obs, next_obs=ns.obs, reward=ns.reward, done=ns.done,
                  truncation=ns.info["truncation"], traj_id=ns.info["traj_id"], action=a)
        return (ns, k), tr

    (_, _), data = jax.jit(lambda s, k: jax.lax.scan(step, (s, k), (), length=steps))(state, key)
    return jax.device_get(data)  # each [steps, num_envs, ...]


def windows(data, length, n, rng):
    T = data["obs"].shape[0]
    out = []
    for _ in range(n):
        e = rng.integers(data["obs"].shape[1])
        s = rng.integers(0, T - length)
        out.append({k: v[s:s + length, e] for k, v in data.items()})
    return {k: np.stack([w[k] for w in out]) for k in out[0]}


def fact_ab(name, num_envs, steps, seed):
    raw = create_env(name)
    sd, gi = raw.state_dim, np.asarray(raw.goal_indices)
    # pass 1: goal distances under random actions, to choose an informative threshold
    d = rollout(raw, num_envs, 300, seed)
    goal = d["obs"][..., sd:]
    thresh = float(np.median(np.linalg.norm(d["next_obs"][..., gi] - goal, axis=-1)))
    raw2 = create_env(name)
    orig_thresh = raw2.goal_reach_thresh
    raw2.goal_reach_thresh = thresh
    d = rollout(raw2, num_envs, steps, seed + 1)
    trunc, done = d["truncation"] > 0, d["done"] > 0

    # Fact A
    obs_next_step = d["obs"][1:]
    t_idx = np.argwhere(trunc[:-1])
    eq_next = [np.allclose(d["next_obs"][t, e], obs_next_step[t, e]) for t, e in t_idx]
    newtraj = [d["traj_id"][t + 1, e] == d["traj_id"][t, e] + 1 for t, e in t_idx]
    # is next_obs at truncation different from what the episode would continue with? compare to obs[t]
    # (auto-reset puts the agent back at the start; distance from the pre-step state is large)
    jump = [float(np.linalg.norm(d["next_obs"][t, e, gi] - d["obs"][t, e, gi])) for t, e in t_idx]
    nontrunc = np.argwhere(~trunc[:-1] & ~done[:-1])[:5000]
    eq_nontrunc = np.mean([np.allclose(d["next_obs"][t, e], obs_next_step[t, e]) for t, e in nontrunc])
    typical_step = [float(np.linalg.norm(d["next_obs"][t, e, gi] - d["obs"][t, e, gi])) for t, e in nontrunc]

    # Fact B
    keep = ~trunc & ~done
    goal = d["obs"][..., sd:]
    r_next = (np.linalg.norm(d["next_obs"][..., gi] - goal, axis=-1) < thresh).astype(np.float32)
    r_cur = (np.linalg.norm(d["obs"][..., gi] - goal, axis=-1) < thresh).astype(np.float32)
    r = d["reward"]
    disagree = keep & (r_next != r_cur)
    return dict(
        env=name,
        state_dim=int(sd), goal_indices=gi.tolist(), goal_reach_thresh=float(orig_thresh),
        A_num_truncation_steps=int(len(t_idx)),
        A_frac_next_obs_equals_following_obs=float(np.mean(eq_next)) if eq_next else None,
        A_frac_following_step_new_traj_id=float(np.mean(newtraj)) if newtraj else None,
        A_mean_goal_jump_at_truncation=float(np.mean(jump)) if jump else None,
        A_mean_goal_step_non_truncation=float(np.mean(typical_step)),
        A_frac_next_obs_equals_following_obs_non_truncation=float(eq_nontrunc),
        B_test_threshold=thresh,
        B_num_steps=int(keep.sum()), B_frac_positive=float(r[keep].mean()),
        B_match_rate_next_obs=float((r[keep] == r_next[keep]).mean()),
        B_match_rate_current_obs=float((r[keep] == r_cur[keep]).mean()),
        B_num_steps_where_conventions_differ=int(disagree.sum()),
        B_match_next_where_differ=float((r[disagree] == r_next[disagree]).mean()) if disagree.any() else None,
        B_match_current_where_differ=float((r[disagree] == r_cur[disagree]).mean()) if disagree.any() else None,
        num_terminations=int(done.sum() - trunc.sum()),
    )


def fact_c(name, num_envs, steps, seed, n_windows=256):
    raw = create_env(name)
    d = rollout(raw, num_envs, steps, seed + 2)
    rng = np.random.default_rng(seed)
    w = windows(d, 1001, n_windows, rng)
    tr = Transition(
        observation=jnp.asarray(w["obs"]), next_observation=jnp.asarray(w["next_obs"]),
        action=jnp.asarray(w["action"]), reward=jnp.asarray(w["reward"]), discount=jnp.asarray(1.0 - w["done"]),
        extras={"state_extras": {"truncation": jnp.asarray(w["truncation"]), "traj_id": jnp.asarray(w["traj_id"])},
                "policy_extras": {}},
    )
    out = {}
    for strat in ("final", "future"):
        cfg = SAC(use_her=True, her_strategy=strat, her_ratio=0.8)
        f = jax.vmap(lambda t, k: flatten_batch(cfg, raw, t, k))
        o = jax.device_get(f(tr, jax.random.split(jax.random.PRNGKey(seed), n_windows)))
        rel = o.extras["her_relabeled"] > 0
        pos = o.reward > 0
        trunc = w["truncation"] > 0
        out[strat] = dict(
            n_transitions=int(rel.size),
            frac_relabelled=float(rel.mean()),
            n_relabelled_reward1=int((rel & pos).sum()),
            n_relabelled_reward1_at_truncation=int((rel & pos & trunc).sum()),
            n_reward1_total=int(pos.sum()),
            n_reward1_not_masked=int((pos & ~trunc).sum()),
            frac_reward1_among_used_by_sac=float((pos & ~trunc).sum() / max((~trunc).sum(), 1)),
        )
    out["stored_env_reward1"] = int((w["reward"] > 0).sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["ant_big_maze", "humanoid", "arm_push_easy"])
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for e in a.envs:
        t = time.time()
        res[e] = {"AB": fact_ab(e, a.num_envs, a.steps, a.seed), "C": fact_c(e, a.num_envs, a.steps, a.seed)}
        res[e]["seconds"] = time.time() - t
        print(json.dumps({e: res[e]}, indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
