"""Learner update, checkpoint compatibility, and unified training smoke tests."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax import envs
from brax.io import model
from brax.training import types
from jaxgcrl.agents import CRL, PPO, SAC, TD3
from jaxgcrl.agents.crl.crl import load_params as load_crl_params
from jaxgcrl.agents.sac.sac import Transition as OffPolicyTransition
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.train import train
from jaxgcrl.utils.config import RunConfig
from jaxgcrl.utils.env import create_env
from orbax import checkpoint as ocp


def _tree_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True)
    )


@pytest.mark.parametrize("name", ["crl", "sac", "td3", "ppo"])
def test_two_updates_preserve_layout_config_and_inference(name):
    env = create_env("reacher")
    config = RunConfig(
        env="reacher",
        num_envs=2,
        episode_length=5,
        total_env_steps=64,
        num_eval_envs=2,
    )
    common = dict(batch_size=2, unroll_length=4)
    if name == "ppo":
        agent = PPO(**common, learning_rate=7e-4, num_minibatches=1, num_updates_per_batch=2)
    elif name == "crl":
        agent = CRL(
            **common,
            h_dim=16,
            n_hidden=2,
            min_replay_size=8,
            max_replay_size=64,
            repr_dim=8,
            skip_connections=0,
            policy_lr=7e-4,
            energy_fn="cosine",
            contrastive_loss_fn="binary_nce",
        )
    elif name == "sac":
        agent = SAC(
            **common,
            h_dim=16,
            n_hidden=2,
            min_replay_size=8,
            max_replay_size=64,
            learning_rate=7e-4,
            normalize_observations=True,
        )
    else:
        agent = TD3(
            **common,
            h_dim=16,
            n_hidden=2,
            min_replay_size=8,
            max_replay_size=64,
            learning_rate=7e-4,
            exploration_noise=0.13,
            policy_delay=3,
        )

    learner = agent.create(0, env.observation_size, env.action_size, config, env)
    assert learner.config is agent
    obs = jnp.ones((2, env.observation_size))
    key = jax.random.PRNGKey(17)
    assert learner.sample_actions(obs, key, True).shape == (2, env.action_size)

    before = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), learner.state)
    if name == "crl":
        from jaxgcrl.agents.crl.crl import Transition as CRLTransition
        from jaxgcrl.agents.crl.crl import flatten_batch

        assert "log_logit_scale" in learner.state.critic_state.params
        raw = CRLTransition(
            jnp.ones((5, env.observation_size)),
            jnp.ones((5, env.action_size)) * 0.1,
            jnp.zeros(5),
            jnp.ones(5),
            {"state_extras": {"traj_id": jnp.zeros(5), "truncation": jnp.zeros(5)}},
        )
        data = flatten_batch(
            (agent.discounting, env.state_dim, tuple(int(i) for i in env.goal_indices)),
            raw,
            key,
        )
        batch = jax.tree_util.tree_map(lambda x: x[:2], data)
    elif name == "ppo":
        batch = types.Transition(
            observation=jnp.ones((1, 2, 4, env.observation_size)),
            action=jnp.zeros((1, 2, 4, env.action_size)),
            reward=jnp.ones((1, 2, 4)),
            discount=jnp.ones((1, 2, 4)),
            next_observation=jnp.ones((1, 2, 4, env.observation_size)),
            extras={
                "state_extras": {"truncation": jnp.zeros((1, 2, 4))},
                "policy_extras": {
                    "raw_action": jnp.zeros((1, 2, 4, env.action_size)),
                    "log_prob": jnp.zeros((1, 2, 4)),
                },
            },
        )
    else:
        batch = OffPolicyTransition(
            jnp.ones((1, 2, env.observation_size)),
            jnp.ones((1, 2, env.observation_size)),
            jnp.ones((1, 2, env.action_size)) * 0.1,
            jnp.ones((1, 2)),
            jnp.ones((1, 2)),
            {"state_extras": {"truncation": jnp.zeros((1, 2))}},
        )

    layouts = [x.shape for x in jax.tree_util.tree_leaves(learner.state)]
    for step in range(2):
        learner, metrics = learner.update(batch, jax.random.fold_in(key, step))
        assert [x.shape for x in jax.tree_util.tree_leaves(learner.state)] == layouts
        assert metrics
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(metrics))
        actions = learner.sample_actions(obs, key, True)
        assert np.isfinite(actions).all()
        assert actions.shape == (2, env.action_size)

    assert any(
        not np.array_equal(x, y)
        for x, y in zip(
            jax.tree_util.tree_leaves(before),
            jax.tree_util.tree_leaves(learner.state),
            strict=True,
        )
    )
    if name != "ppo":
        np.testing.assert_equal(np.asarray(learner.state.gradient_steps), 2)

    params = learner.inference_params()
    np.testing.assert_allclose(
        learner.make_policy(params, deterministic=True)(obs, key)[0],
        learner.sample_actions(obs, key, True),
        atol=0,
        rtol=0,
    )


def test_td3_delayed_policy_and_configured_collection_noise():
    env = create_env("reacher")
    config = RunConfig(env="reacher", num_envs=2, episode_length=5, max_devices_per_host=1)
    agent = TD3(
        h_dim=16,
        n_hidden=2,
        batch_size=2,
        policy_delay=3,
        exploration_noise=0.13,
        noise_clip=0.17,
    )
    learner = agent.create(0, env.observation_size, env.action_size, config, env)
    obs = jnp.ones((2, env.observation_size))
    key = jax.random.PRNGKey(11)

    deterministic = learner.sample_actions(obs, key, True)
    stochastic = learner.sample_actions(obs, key, False)
    noise = jnp.clip(jax.random.normal(key, deterministic.shape) * 0.13, -0.17, 0.17)
    np.testing.assert_array_equal(stochastic, deterministic + noise)

    batch = OffPolicyTransition(
        obs[None],
        (obs * 0.9)[None],
        jnp.ones((1, 2, env.action_size)) * 0.1,
        jnp.ones((1, 2)),
        jnp.ones((1, 2)),
        {"state_extras": {"truncation": jnp.zeros((1, 2))}},
    )
    first, _ = learner.update(batch, key)
    assert not _tree_equal(learner.state.policy_params, first.state.policy_params)

    second, _ = first.update(batch, key)
    assert _tree_equal(first.state.policy_params, second.state.policy_params)
    assert _tree_equal(first.state.target_policy_params, second.state.target_policy_params)
    assert _tree_equal(first.state.target_q_params, second.state.target_q_params)
    assert not _tree_equal(first.state.q_params, second.state.q_params)


@pytest.mark.parametrize("agent_cls,transform", [(SAC, "pmap"), (TD3, "vmap")])
def test_reset_mapping_preserves_legacy_transform(agent_cls, transform):
    env = envs.training.wrap(
        TrajectoryIdWrapper(create_env("reacher")),
        episode_length=5,
        action_repeat=1,
    )
    devices = tuple(jax.local_devices()[:1])
    keys = jax.random.split(jax.random.PRNGKey(19), 2).reshape(1, 2, 2)
    expected = (
        jax.pmap(env.reset, devices=devices)(keys) if transform == "pmap" else jax.vmap(env.reset)(keys)
    )
    agent = agent_cls(h_dim=16, n_hidden=2)
    actual = agent.reset_training_env(env, keys, devices)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for got, wanted in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(got, wanted)


def test_ppo_zero_budget_and_orbax_restore(tmp_path):
    env = create_env("reacher")
    config = RunConfig(
        env="reacher",
        total_env_steps=0,
        num_envs=2,
        num_eval_envs=2,
        episode_length=5,
        num_evals=2,
    )
    agent = PPO(batch_size=2, num_minibatches=1, num_updates_per_batch=2, unroll_length=4)
    calls = []
    make_policy, params, metrics = train(agent, env, env, config, lambda *args, **kwargs: calls.append(args))
    assert calls == []
    assert metrics == {}
    assert not hasattr(params[1], "value")
    actions, _ = make_policy(params, deterministic=True)(
        jnp.ones((2, env.observation_size)), jax.random.PRNGKey(0)
    )
    assert actions.shape == (2, env.action_size)
    assert np.isfinite(actions).all()

    learner = agent.create(0, env.observation_size, env.action_size, config, env)
    state = jax.tree_util.tree_map(lambda x: x[0], learner.state)
    ckpt_path = str(tmp_path / "ppo-init")
    ocp.PyTreeCheckpointer().save(ckpt_path, (state.normalizer_params, state.params))
    restored = dataclasses.replace(agent, restore_checkpoint_path=ckpt_path).create(
        5, env.observation_size, env.action_size, config, env
    )
    for expected, actual in zip(
        jax.tree_util.tree_leaves(learner.state.params),
        jax.tree_util.tree_leaves(restored.state.params),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("name", ["crl", "sac", "td3", "ppo"])
def test_unified_train_short_run_and_checkpoint_roundtrip(name, tmp_path):
    env_name = "ant" if name == "crl" else "reacher"
    env = create_env(env_name)
    ckpt_dir = tmp_path / name
    ckpt_dir.mkdir()

    if name == "crl":
        config = RunConfig(
            env=env_name,
            total_env_steps=32,
            num_envs=2,
            num_eval_envs=2,
            episode_length=5,
            num_evals=1,
            checkpoint_logdir=str(ckpt_dir),
        )
        agent = CRL(
            batch_size=2,
            unroll_length=4,
            min_replay_size=8,
            max_replay_size=64,
            h_dim=16,
            n_hidden=2,
            repr_dim=8,
            skip_connections=0,
        )
    elif name == "ppo":
        config = RunConfig(
            env=env_name,
            total_env_steps=16,
            num_envs=2,
            num_eval_envs=2,
            episode_length=5,
            num_evals=2,
        )
        agent = PPO(
            batch_size=2,
            num_minibatches=1,
            num_updates_per_batch=1,
            unroll_length=4,
        )
    elif name == "sac":
        config = RunConfig(
            env=env_name,
            total_env_steps=24,
            num_envs=2,
            num_eval_envs=2,
            episode_length=5,
            num_evals=2,
            checkpoint_logdir=str(ckpt_dir / "ckpt"),
        )
        agent = SAC(
            batch_size=2,
            unroll_length=4,
            min_replay_size=4,
            max_replay_size=64,
            h_dim=16,
            n_hidden=2,
            use_her=True,
        )
    else:
        config = RunConfig(
            env=env_name,
            total_env_steps=24,
            num_envs=2,
            num_eval_envs=2,
            episode_length=5,
            num_evals=2,
            checkpoint_logdir=str(ckpt_dir / "ckpt"),
        )
        agent = TD3(
            batch_size=2,
            unroll_length=4,
            min_replay_size=4,
            max_replay_size=64,
            h_dim=16,
            n_hidden=2,
            use_her=True,
        )

    make_policy, params, metrics = train(agent, env, env, config)
    assert params is not None
    assert "training/sps" in metrics

    obs = jnp.ones((2, env.observation_size))
    key = jax.random.PRNGKey(3)
    expected_actions, _ = make_policy(params, deterministic=True)(obs, key)
    assert expected_actions.shape == (2, env.action_size)
    assert np.isfinite(expected_actions).all()

    final_path = tmp_path / f"{name}_final.pkl"
    model.save_params(str(final_path), params)
    reloaded = model.load_params(str(final_path))
    reloaded_actions, _ = make_policy(reloaded, deterministic=True)(obs, key)
    np.testing.assert_array_equal(reloaded_actions, expected_actions)

    if name == "crl":
        periodic = next(ckpt_dir.glob("step_*.pkl"))
        periodic_tuple = load_crl_params(str(periodic))
        assert isinstance(periodic_tuple, tuple) and len(periodic_tuple) == 3
        periodic_actions, _ = make_policy(periodic_tuple, deterministic=True)(obs, key)
        np.testing.assert_array_equal(periodic_actions, expected_actions)
    elif name in ("sac", "td3"):
        periodic = next(ckpt_dir.glob(f"ckpt_{name}_*.pkl"))
        periodic_params = model.load_params(str(periodic))
        periodic_actions, _ = make_policy(periodic_params, deterministic=True)(obs, key)
        np.testing.assert_array_equal(periodic_actions, expected_actions)


def test_multidevice_learner_updates_match_vmap_reference():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import jax
import jax.numpy as jnp
import numpy as np
from brax.training import types
from jaxgcrl.agents import PPO, SAC, TD3
from jaxgcrl.agents.sac.sac import Transition
from jaxgcrl.utils.config import RunConfig
from jaxgcrl.utils.env import create_env

devices = tuple(jax.local_devices())
assert len(devices) >= 2
env = create_env("reacher")
config = RunConfig(env="reacher", num_envs=2, max_devices_per_host=2, episode_length=5)

for name in ("sac", "td3", "ppo"):
    agent = (
        PPO(batch_size=2, num_minibatches=1, num_updates_per_batch=2, unroll_length=4)
        if name == "ppo"
        else {"sac": SAC, "td3": TD3}[name](batch_size=2, h_dim=16, n_hidden=2)
    )
    learner = agent.create(0, env.observation_size, env.action_size, config, env)
    assert learner.devices == devices[:2]
    one = agent.create(
        0,
        env.observation_size,
        env.action_size,
        config.replace(max_devices_per_host=1),
        env,
    )
    assert one.devices == devices[:1]
    assert all(x.shape[0] == 1 for x in jax.tree_util.tree_leaves(one.state))

    obs = jnp.ones((2, 2, env.observation_size)).at[1].multiply(0.7)
    if name == "ppo":
        obs = obs[:, :1]
        batch = types.Transition(
            observation=jnp.repeat(obs[:, :, None, :], 4, axis=2),
            action=jnp.zeros((2, 1, 4, env.action_size)),
            reward=jnp.ones((2, 1, 4)),
            discount=jnp.ones((2, 1, 4)),
            next_observation=jnp.repeat(obs[:, :, None, :], 4, axis=2),
            extras={
                "state_extras": {"truncation": jnp.zeros((2, 1, 4))},
                "policy_extras": {
                    "raw_action": jnp.zeros((2, 1, 4, env.action_size)),
                    "log_prob": jnp.zeros((2, 1, 4)),
                },
            },
        )
    else:
        batch = Transition(
            obs,
            obs * 0.9,
            jnp.ones((2, 2, env.action_size)) * 0.1,
            jnp.ones((2, 2)),
            jnp.ones((2, 2)),
            {"state_extras": {"truncation": jnp.zeros((2, 2))}},
        )

    for step in range(2):
        key = jax.random.PRNGKey(step + 10)
        keys = jax.random.split(key, 2)
        reference = jax.jit(
            jax.vmap(lambda state, data, rng: learner.update_step((state, rng), data), axis_name="i")
        )
        (expected, _), _ = reference(learner.state, batch, keys)
        learner, metrics = learner.update(batch, key)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(metrics))
        for actual, target in zip(
            jax.tree_util.tree_leaves(learner.state),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            assert actual.shape[0] == 2
            np.testing.assert_allclose(actual, target, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(actual[0], actual[1], atol=0, rtol=0)
"""
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = f"{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    existing_xla = os.environ.get("XLA_FLAGS", "")
    xla_flags = f"{existing_xla} --xla_force_host_platform_device_count=2".strip()
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "JAX_PLATFORMS": "cpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "float32",
        "XLA_FLAGS": xla_flags,
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "MPLBACKEND": "Agg",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
