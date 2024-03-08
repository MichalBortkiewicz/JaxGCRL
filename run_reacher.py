import functools
from datetime import datetime

from matplotlib import pyplot as plt

from crl_new.train import train
from reacher import Reacher

if __name__ == "__main__":

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


    class MetricsRecorder:
        def __init__(self):
            self.x_data = []
            self.y_data = {}
            self.y_data_err = {}
            self.times = [datetime.now()]

            self.max_x, self.min_x = num_timesteps * 1.1, 0

        def record(self, num_steps, metrics):
            self.times.append(datetime.now())
            self.x_data.append(num_steps)

            for key, value in metrics.items():
                if key not in self.y_data:
                    self.y_data[key] = []
                    self.y_data_err[key] = []

                self.y_data[key].append(value)
                self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

        def plot_progress(self):
            num_plots = len(self.y_data)
            num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

            fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

            for idx, (key, y_values) in enumerate(self.y_data.items()):
                row = idx // 2
                col = idx % 2

                print(
                    f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}"
                )

                axs[row, col].set_xlim(self.min_x, self.max_x)
                axs[row, col].set_xlabel("# environment steps")
                axs[row, col].set_ylabel(key)
                axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
                axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

            # Hide any empty subplots
            for idx in range(num_plots, num_rows * 2):
                row = idx // 2
                col = idx % 2
                axs[row, col].axis("off")
            plt.tight_layout()
            plt.show()

        def print_times(self):
            print(f"time to jit: {self.times[1] - self.times[0]}")
            print(f"time to train: {self.times[-1] - self.times[1]}")


    metrics_recorder = MetricsRecorder()

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
        metrics_recorder.print_times()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
