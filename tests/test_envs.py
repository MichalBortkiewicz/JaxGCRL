import jax
import pytest

from jaxgcrl.utils.env import create_env

# Sanity check tests for all environments
# python -m pytest to run them


ENVS = [
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "ant",
    "ant_u_maze",
    "ant_big_maze",
    "ant_push",
    "ant_ball",
    "ant_ball_u_maze",
    "ant_ball_big_maze",
    "humanoid",
    "arm_reach",
    "arm_grasp",
    "arm_push_easy",
    "arm_push_hard",
    "arm_binpick_easy",
    "arm_binpick_hard",
]


@pytest.mark.parametrize("env_name", ["aaaa", "", "fake_env"])
def test_error_on_wrong_environment(env_name):
    with pytest.raises(ValueError, match=f"Unknown environment: {env_name}"):
        create_env(env_name=env_name)


def test_error_on_wrong_maze():
    with pytest.raises(ValueError, match=f"Unknown maze layout: fake_maze"):
        create_env(env_name="ant_fake_maze")


@pytest.mark.parametrize("env_name", ENVS)
class TestEnvironment:
    def test_initialization(self, env_name):
        create_env(env_name=env_name)

    def test_environment_has_attributes(self, env_name):
        env = create_env(env_name=env_name)

        assert hasattr(env, "state_dim")
        assert hasattr(env, "goal_indices")

    def test_environment_reset(self, env_name):
        env = create_env(env_name=env_name)
        jit_env_reset = jax.jit(env.reset)
        rng = jax.random.PRNGKey(seed=0)

        state = jit_env_reset(rng=rng)

        assert not jax.numpy.isnan(state.obs).any()
        assert not jax.numpy.isinf(state.obs).any()

    def test_environment_step(self, env_name):
        env = create_env(env_name=env_name)
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        rng = jax.random.PRNGKey(seed=0)
        action_size = env.action_size

        state = jit_env_reset(rng=rng)

        for _ in range(100):
            act_rng, rng = jax.random.split(rng)
            act = jax.random.uniform(act_rng, (action_size,))
            state = jit_env_step(state, act)

            assert not jax.numpy.isnan(state.obs).any()
            assert not jax.numpy.isinf(state.obs).any()
