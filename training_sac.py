import functools
import json
import os

import jax
import wandb
from brax.io import model
from brax.io import html
from brax.training.agents.sac.train import train
from pyinstrument import Profiler


from utils import MetricsRecorder, create_env, get_tested_args, create_parser


def render(inf_fun_factory, params, env, exp_dir, exp_name):
    inference_fn = inf_fun_factory(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for i in range(5000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        if i % 1000 == 0:
            state = jit_env_reset(rng=rng)

    url = html.render(env.sys.replace(dt=env.dt), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})


def main(args):

    env = create_env(args)

    train_fn = functools.partial(
        train,
        num_timesteps=6_553_600,
        num_evals=50,
        reward_scaling=30,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        max_devices_per_host=1,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=args.seed,
    )

    metrics_recorder = MetricsRecorder(args.num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0

    metrics_to_collect = [
        "eval/episode_reward",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_ctrl",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/crl_critic_loss",
        "training/actor_loss",
        "training/critic_loss",
        "training/binary_accuracy",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
        "training/sps",
        "training/entropy",
        "training/alpha",
        "training/alpha_loss",
        "training/entropy",
    ]

    def progress(num_steps, metrics):
        for key in metrics_to_collect:
            ensure_metric(metrics, key)
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        metrics_recorder.log_wandb()
        metrics_recorder.print_progress()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    os.makedirs("./params", exist_ok=True)
    model.save_params(f'./params/param_{args.exp_name}_s_{args.seed}', params)
    render(make_inference_fn, params, env, "./renders", args.exp_name)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.use_tested_args:
        args = get_tested_args(args)

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )
    sgd_to_env = (
        args.num_envs
        * args.episode_length
        * args.multiplier_num_sgd_steps
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    print(f"SGD steps per env steps: {sgd_to_env}")
    args.sgd_to_env = sgd_to_env

    wandb.init(
        project="crl",
        name=args.exp_name,
        config=vars(args),
        mode="online" if args.log_wandb else "disabled",
    )

    with Profiler(interval=0.1) as profiler:
        main(args)
    profiler.print()
    profiler.open_in_browser()
