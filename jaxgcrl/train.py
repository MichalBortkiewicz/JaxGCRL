"""Public training entry point and collection-strategy dispatch."""

from jaxgcrl.training import episode_off_policy, off_policy, on_policy

STRATEGIES = {
    "off_policy": off_policy.run,
    "episode_off_policy": episode_off_policy.run,
    "on_policy": on_policy.run,
}


def train(
    agent,
    train_env,
    eval_env=None,
    config=None,
    progress_fn=lambda *args, **kwargs: None,
    randomization_fn=None,
):
    if config is None:
        raise ValueError("A run configuration is required")
    try:
        strategy = STRATEGIES[agent.strategy]
    except KeyError as error:
        raise ValueError(f"Unsupported collection strategy: {agent.strategy}") from error
    return strategy(
        agent,
        config,
        train_env,
        train_env if eval_env is None else eval_env,
        randomization_fn,
        progress_fn,
    )
