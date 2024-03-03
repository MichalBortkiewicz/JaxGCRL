import functools
from datetime import datetime

from brax import envs
from matplotlib import pyplot as plt

from crl_new.train import train

if __name__ == "__main__":
    env_name = 'reacher'
    env = envs.get_environment(env_name)

    #%%
    train_fn = functools.partial(
        train,
        num_timesteps=1000000,
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
        batch_size=32,
        seed=0,
    )


    x_data = []
    y_data = []
    y_data2 = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 0, -50


    def progress(num_steps, metrics):
        print(metrics.keys())
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        if "eval/crl_critic_loss" in metrics.keys():
            y_data2.append(metrics["eval/crl_critic_loss"])
        else:
            y_data2.append(0)
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))

        print(f"step: {num_steps}, reward: {y_data[-1]:.3f} +/- {ydataerr[-1]:.3f}")
        axs[0].set_xlim(0, train_fn.keywords["num_timesteps"] * 1.1)
        axs[0].set_ylim(min_y, max_y)
        axs[0].set_xlabel("# environment steps")
        axs[0].set_ylabel("reward per episode")
        axs[0].errorbar(x_data, y_data, yerr=ydataerr)
        axs[0].set_title(f"y={y_data[-1]:.3f}")

        axs[1].set_xlim(0, train_fn.keywords["num_timesteps"] * 1.1)
        axs[1].set_xlabel("# environment steps")
        axs[1].set_ylabel("crl_critic_loss")
        axs[1].plot(x_data, y_data2)
        axs[1].set_title(f"CRL:y={y_data2[-1]:.3f}")

        plt.tight_layout()
        plt.show()
        print(f"CRL loss: {y_data2[-1]}")


    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")