import os
import pandas as pd
import numpy as np

# Function to extract metrics from runs
def extract_run_data(run, download_again:bool=False):
    # Extract run config and metrics
    config = run.config
    summary = run.summary._json_dict
    # TODO: potentially just load csv
    if os.path.exists(f"./data/history_{run.id}.csv") and not download_again:
        history = pd.read_csv(f"./data/history_{run.id}.csv")
    else:
        history = run.history(pandas=True)
        if history.shape[0] != 50:
            raise Exception(f"History shape[0] is not 50. history.shape=={history.shape}. Run: {run}")
        history.to_csv(f"./data/history_{run.id}.csv", index=False)
    # Combine run information into a single dictionary

    run_data = {"run_id": run.id, "name": run.name, **config, **summary}
    return run_data, history


def create_rliable_compatible_data(df, metric):
    return df[metric].to_numpy()


def moving_average_smoothing(data, window_size=5):
    """Apply a moving average filter to the last axis of the input data, ensuring no wrap-around."""
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode="edge")
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def aggregate_data_from_wandb(
    runs,
    metric: str,
    exp_names: list,
    exp_names_mapping: dict,
    env_title_mapping: dict,
    take_x_seeds: int,
    single_env=False,
    download_again: bool = False,
):
    # Download/load history from wandb
    all_histories = []
    for run in runs:
        run_data, history = extract_run_data(run, download_again)
        all_histories.append(history)

    # Create first data structure: dict of dict of list (methods->envs->seeds).
    if single_env:
        data = {exp_names_mapping[elem]: {env_title_mapping[single_env]: []} for elem in exp_names}
        seeds = {exp_names_mapping[elem]: {env_title_mapping[single_env]: []} for elem in exp_names}
    else:
        data = {
            exp_names_mapping[elem]: {elem_inner: [] for elem_inner in env_title_mapping.values()} for elem in exp_names
        }
        seeds = {
            exp_names_mapping[elem]: {elem_inner: [] for elem_inner in env_title_mapping.values()} for elem in exp_names
        }

    for run, history in zip(runs, all_histories):
        a = create_rliable_compatible_data(history, metric)
        if (
            len(seeds[exp_names_mapping[run.config["exp_name"]]][env_title_mapping[run.config["env_name"]]])
            >= take_x_seeds
        ):
            continue

        # Make sure to not take duplicated seeds
        if (
            run.config["seed"]
            in seeds[exp_names_mapping[run.config["exp_name"]]][env_title_mapping[run.config["env_name"]]]
        ):
            print(f"Dropping for {run.config['exp_name']}")
            continue

        data[exp_names_mapping[run.config["exp_name"]]][env_title_mapping[run.config["env_name"]]].append(a)
        seeds[exp_names_mapping[run.config["exp_name"]]][env_title_mapping[run.config["env_name"]]].append(
            run.config["seed"]
        )

    # Create intermediate data structure: dict of dict of arrays
    # For instance data['L2']['Ant Ball'].shape == num_seeds x 1 x timesteps (50)

    if single_env:
        for method in exp_names_mapping.values():
            data[method][env_title_mapping[single_env]] = np.array(data[method][env_title_mapping[single_env]])[
                :, None, :
            ]
    else:
        for env in env_title_mapping.values():
            for method in exp_names_mapping.values():
                # print(f"{[len(elem) for elem in data[method][env]]}")
                data[method][env] = np.array(data[method][env])[:, None, :]

    # just to verify
    if single_env:
        for method in exp_names_mapping.values():
            print(f"{method}, {single_env}")
            print(data[method][env_title_mapping[single_env]].shape)
    else:
        for env in env_title_mapping.values():
            for method in exp_names_mapping.values():
                print(f"{method}, {env}")
                print(data[method][env].shape)

    # Create final data structure dict of arrays (methods->array)
    # array shape: num_seeds x num_envs x timesteps
    data_new = {key: np.concatenate((list(data[key].values())), axis=1) for key, elem in data.items()}

    return data_new