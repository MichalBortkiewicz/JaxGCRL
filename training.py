import functools
import wandb

from crl.train import train
from envs.reacher import Reacher
from utils import MetricsRecorder

if __name__ == "__main__":

    wandb.init(project="crl", name="reacher-test")

    env = Reacher()

    num_timesteps = 1000000
    train_fn = functools.partial(
        train,
        num_timesteps=num_timesteps,
        max_replay_size=100000,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=50,
        normalize_observations=True,
        action_repeat=1,
        grad_updates_per_step=2,
        discounting=0.97,
        learning_rate=3e-4,
        # For debug purposes
        num_envs=2048,
        batch_size=512,
        seed=0,
        unroll_length=50
    )


    metrics_recorder = MetricsRecorder(num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0

    metrics_to_collect = [
        "eval/episode_reward",
        "training/crl_critic_loss",
        "training/critic_loss",
        "training/crl_actor_loss",
        "training/actor_loss",
    ]

    def progress(num_steps, metrics):
        for key in metrics_to_collect:
            ensure_metric(metrics, key)
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        metrics_recorder.plot_progress()
        metrics_recorder.log_wandb()
        # metrics_recorder.print_times()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
