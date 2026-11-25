"""Learner contracts and device-placement helpers shared across training strategies."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import jax
import jax.numpy as jnp


class AgentConfig(Protocol):
    strategy: str

    def create(self, seed, obs_size, action_size, config, env, *, key=None, devices=None): ...
    def update(self, learner, batch, key): ...
    def sample_actions(self, learner, obs, key, deterministic=False): ...


@dataclass(frozen=True)
class Learner:
    config: AgentConfig
    state: Any
    update_step: Callable
    make_policy: Callable
    export: Callable
    devices: tuple[Any, ...] = ()
    network: Any = None
    prepare_batch: Any = None

    @property
    def replicated(self):
        return bool(self.devices)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def inference_params(self):
        state = self.state
        if self.replicated:
            state = jax.tree_util.tree_map(lambda x: x[0], state)
        return self.export(state)

    def update(self, batch, key):
        return self.config.update(self, batch, key)

    def sample_actions(self, obs, key, deterministic=False):
        return self.config.sample_actions(self, obs, key, deterministic)


def update_minibatch(learner, batch, key):
    """Apply one learner update step while retaining replicated state layout."""
    if learner.replicated:
        keys = local_update_keys(key, len(learner.devices))
        apply = jax.pmap(
            lambda state, data, rng: learner.update_step((state, rng), data),
            axis_name="i",
            devices=collective_devices(learner.devices),
        )
        (state, _), metrics = apply(learner.state, batch, keys)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    else:
        (state, _), metrics = learner.update_step((learner.state, key), batch)
    return learner.replace(state=state), metrics


def selected_devices(config):
    """Select local placement devices while validating the global participant topology."""
    maximum = config.max_devices_per_host
    if maximum is not None and maximum < 1:
        raise ValueError("max_devices_per_host must be positive")
    devices = tuple(jax.local_devices())
    devices = devices if maximum is None else devices[:maximum]
    collective_devices(devices)
    return devices


def collective_devices(local_devices):
    """Return a consistent process-major pmap participant list on every host."""
    local_devices = tuple(local_devices)
    if not local_devices:
        raise ValueError("at least one local device is required")
    process_count = jax.process_count()
    if process_count == 1:
        return local_devices
    hosts = [tuple(d for d in jax.devices() if d.process_index == p) for p in range(process_count)]
    counts = {len(host) for host in hosts}
    if len(counts) != 1 or not all(hosts):
        raise ValueError("equal nonempty device topology is required on every process")
    count = len(local_devices)
    if local_devices != hosts[jax.process_index()][:count]:
        raise ValueError("selected devices must be a local prefix shared across processes")
    return tuple(device for host in hosts for device in host[:count])


def local_update_keys(key, count):
    """Split distinct per-device update keys without altering single-process RNG."""
    if jax.process_count() > 1:
        key = jax.random.fold_in(key, jax.process_index())
    return jax.random.split(key, count)
