import jax
from brax.io import model
from brax.io import html
from crl import networks as sac_networks
from envs.reacher import Reacher

network_factory = sac_networks.make_sac_networks
normalize_fn = lambda x, y: x
sac_network = network_factory(
    observation_size=20,
    action_size=2,
    preprocess_observations_fn=normalize_fn,
)
make_inference_fn = sac_networks.make_inference_fn(sac_network)

params = model.load_params('./tmp/params')
inference_fn = make_inference_fn(params)

if True:
    env = Reacher()

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for _ in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)

    url = html.render(env.sys.replace(dt=env.dt), rollout)
    print(url)