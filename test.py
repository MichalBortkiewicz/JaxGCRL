import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from IPython.display import HTML
from brax.io import model, html
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from crl_new import networks
from utils import get_env_config, create_env
import pickle
import numpy as np

from jax import config
# config.update("jax_debug_nans", True)

RUN_FOLDER_PATH = './runs/run_hard_ant_test_s_1'
CKPT_NAME = '/final'

params = model.load_params(RUN_FOLDER_PATH + '/ckpt' + CKPT_NAME)
processor_params, policy_params, encoders_params = params
sa_encoder_params, g_encoder_params = encoders_params['sa_encoder'], encoders_params['g_encoder']

args_path = RUN_FOLDER_PATH + '/args.pkl'

with open(args_path, "rb") as f:
    args = pickle.load(f)


args.env_name = "hard_ant"
args.backend= "spring"
config = get_env_config(args)

env = create_env(args)
obs_size = env.observation_size
action_size = env.action_size

crl_networks = networks.make_crl_networks(config, obs_size, action_size)

inference_fn = networks.make_inference_fn(crl_networks)
inference_fn = inference_fn(params[:2])

sa_encoder = lambda obs: crl_networks.sa_encoder.apply(processor_params, sa_encoder_params, obs)
g_encoder = lambda obs: crl_networks.g_encoder.apply(processor_params, g_encoder_params, obs)

NUM_EPISODES = 5

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)


trajectories = []
rollouts = []

for episode in range(NUM_EPISODES):
    trajectory = []
    rng = jax.random.PRNGKey(seed=episode)
    state = jit_env_reset(rng=rng)
    for _ in range(1000):
        rollouts.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        # print("obs", state.obs)
        # print("state", state.obs[:31])
        # print("goal", state.obs[31:])
        # print("goal_target", state.obs[-4:-2])
        # print("dist", state.metrics['dist'])
        # print("\n\n\n\n\n")
        trajectory.append((state,act))
        state = jit_env_step(state, act)
    trajectories.append(trajectory)

rend = html.render(env.sys.replace(dt=env.dt), rollouts)

with open("renders/env_render.html", "w") as f:
    f.write(rend)

# print(state)