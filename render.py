import os
from collections import namedtuple

import jax
from brax.io import model
from brax.io import html
from crl_new import networks as sac_networks
from envs.reacher import Reacher


def save_html_file(html_code, file_path):
    with open(file_path, 'w') as file:
        file.write(html_code)


def open_html_in_browser(file_path):
    import webbrowser
    webbrowser.open('file://' + file_path)

Config = namedtuple(
    "Config",
    "sac debug discount obs_dim start_index end_index goal_start_idx goal_end_idx goal_dim unroll_length episode_length repr_dim",
)
CONFIG = Config(
    sac=False,
    debug=False,
    discount=0.99,
    obs_dim=10,
    start_index=0,
    end_index=10,
    goal_start_idx=4,
    goal_end_idx=10,
    goal_dim=6,
    unroll_length=50,
    episode_length=50,
    repr_dim=64,
)

network_factory = sac_networks.make_sac_networks
normalize_fn = lambda x, y: x
sac_network = network_factory(
    config=CONFIG,
    observation_size=16,
    action_size=2,
    preprocess_observations_fn=normalize_fn,
)
make_inference_fn = sac_networks.make_inference_fn(sac_network)

params = model.load_params('./params/param_big_test2_s_1')
inference_fn = make_inference_fn(params)
rollout = []
jit_inference_fn = jax.jit(inference_fn)

env = Reacher()
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
for i in range(1, 1000):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
    if i % 50 == 0:
        env = Reacher()
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        rng = jax.random.PRNGKey(seed=i)
        state = jit_env_reset(rng=rng)

url = html.render(env.sys.replace(dt=env.dt), rollout)
print(url)

save_html_file(url, 'render.html')
open_html_in_browser(os.path.join(os. getcwd(),'render.html'))
