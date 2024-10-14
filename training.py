import functools
import json
import os
import pickle

import jax
import math
import wandb
from brax.io import model
from brax.io import html
from pyinstrument import Profiler


from src.train import train
from utils import MetricsRecorder, get_env_config, create_env, create_eval_env, create_parser


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

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})


def main(args):

    env = create_env(args)
    eval_env = create_eval_env(args)
    config = get_env_config(args)


    os.makedirs('./runs', exist_ok=True)
    run_dir = './runs/run_{name}_s_{seed}'.format(name=args.exp_name, seed=args.seed)
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    train_fn = functools.partial(
        train,
        num_timesteps=args.num_timesteps,
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        num_evals=args.num_evals,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        contrastive_loss_fn=args.contrastive_loss_fn,
        energy_fn=args.energy_fn,
        logsumexp_penalty=args.logsumexp_penalty,
        l2_penalty=args.l2_penalty,
        resubs=not args.no_resubs,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        unroll_length=args.unroll_length,
        multiplier_num_sgd_steps=args.multiplier_num_sgd_steps,
        config=config,
        checkpoint_logdir=ckpt_dir,
        eval_env=eval_env,
        use_c_target=args.use_c_target,
        exploration_coef=args.exploration_coef,
        use_ln=args.use_ln,
        h_dim=args.h_dim,
        n_hidden=args.n_hidden,
    )

    metrics_recorder = MetricsRecorder(args.num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is Nan")

    metrics_to_collect = [
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/crl_critic_loss",
        "training/actor_loss",
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
        "training/sa_repr_mean",
        "training/g_repr_mean",
        "training/sa_repr_std",
        "training/g_repr_std",
        "training/c_target",
        "training/l_align",
        "training/l_unif",
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

    model.save_params(ckpt_dir + '/final', params)
    render(make_inference_fn, params, env, run_dir, args.exp_name)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

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
        project=args.project_name,
        group=args.group_name,
        name=args.exp_name,
        config=vars(args),
        mode="online" if args.log_wandb else "disabled",
    )

    with Profiler(interval=0.1) as profiler:
        main(args)
    profiler.print()
    profiler.open_in_browser()
