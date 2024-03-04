import functools
from datetime import datetime

from brax import envs
from matplotlib import pyplot as plt

from crl_new.train import train

if __name__ == "__main__":
    env_name = 'reacher'
    env = envs.get_environment(env_name)

    num_timesteps = 1000000
    train_fn = functools.partial(
        train,
        num_timesteps=num_timesteps,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=50,
        normalize_observations=True,
        action_repeat=1,
        grad_updates_per_step=2,
        discounting=0.97,
        learning_rate=3e-4,
        # For debug purposes
        num_envs=16,
        batch_size=8,
        seed=0,
    )


    class MetricsRecorder:
        def __init__(self):
            self.x_data = []
            self.y_data = {}
            self.y_data_err = {}
            self.times = [datetime.now()]

            self.max_x, self.min_x = 0, num_timesteps*1.1

        def record(self, num_steps, metrics):
            self.times.append(datetime.now())
            self.x_data.append(num_steps)

            for key, value in metrics.items():
                if key not in self.y_data:
                    self.y_data[key] = []
                    self.y_data_err[key] = []

                self.y_data[key].append(value)
                self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

            self.plot_progress()
            self.print_times()

        def plot_progress(self):
            fig, axs = plt.subplots(len(self.y_data), 1, figsize=(10, 5 * len(self.y_data)))

            for idx, (key, y_values) in enumerate(self.y_data.items()):
                print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")

                axs[idx].set_xlim(self.min_x, self.max_x)
                axs[idx].set_xlabel("# environment steps")
                axs[idx].set_ylabel(key)
                axs[idx].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
                axs[idx].set_title(f"{key}: {y_values[-1]:.3f}")

            plt.tight_layout()
            plt.show()

        def print_times(self):
            print(f"time to jit: {self.times[1] - self.times[0]}")
            print(f"time to train: {self.times[-1] - self.times[1]}")


    metrics_recorder = MetricsRecorder()

    def progress(num_steps, metrics):
        if "training/crl_critic_loss" in metrics.keys():
            pass
        else:
            metrics["training/crl_critic_loss"] = 0

        metrics_recorder.record(
            num_steps,
            {
                key: value
                for key, value in metrics.items()
                if key in ["eval/episode_reward", "training/crl_critic_loss"]
            },
        )

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
