"""Training utilities shared by the agents: gradient updates with norms, checkpoint/resume, NaN guard."""

import json
import logging
import os
import pickle
import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax


class NonFiniteError(RuntimeError):
    """Raised when a training loss or metric becomes NaN/Inf. run.py turns it into a FAILED_NAN marker."""


def gradient_update_fn_with_norm(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Same computation as brax.training.gradients.gradient_update_fn (brax 0.12.1), plus the global
    gradient norm (after the cross-device mean). Returns (value, params, optimizer_state, grad_norm)."""
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def f(*args, optimizer_state):
        value, grads = g(*args)
        if pmap_axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=pmap_axis_name)
        grad_norm = optax.global_norm(grads)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state, grad_norm

    return f


def tree_all_finite(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(np.all(np.isfinite(np.asarray(x)))) for x in leaves)


def check_finite_metrics(metrics: dict, keys=("training/critic_loss", "training/actor_loss")):
    bad = {}
    for k in keys:
        if k in metrics:
            v = np.asarray(metrics[k])
            if not np.all(np.isfinite(v)):
                bad[k] = float(np.mean(v))
    if bad:
        raise NonFiniteError(f"non-finite training metrics: {bad}")


class Checkpointer:
    """Atomic save/restore of the full training state, as a flat list of numpy leaves.

    The state is any pytree (training state, env state, replay buffer state, RNG keys). It is saved as
    its leaves only; on restore, the caller passes a freshly built template with the same structure and
    the leaves are put back into it. This avoids pickling custom classes and checks shapes on restore.

    Saves at most every `interval_min` minutes unless `force=True`.
    """

    def __init__(self, run_dir: Optional[str], interval_min: float = 30.0, enabled: bool = True):
        self.enabled = bool(enabled and run_dir)
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "latest.pkl") if run_dir else None
        self.interval_s = interval_min * 60.0
        self._last_save = time.time()

    def exists(self) -> bool:
        return self.enabled and os.path.exists(self.path)

    def save(self, state_tree, meta: dict):
        if not self.enabled:
            return
        t = time.time()
        host = jax.device_get(state_tree)
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(host)]
        payload = {"meta": dict(meta, saved_at=time.strftime("%Y-%m-%d %H:%M:%S")), "leaves": leaves}
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._last_save = time.time()
        size_gb = os.path.getsize(self.path) / 1e9
        logging.info("checkpoint saved: %s (%.2f GB, %.1f s) meta=%s", self.path, size_gb, time.time() - t, meta)

    def maybe_save(self, state_tree, meta: dict, force: bool = False):
        if self.enabled and (force or time.time() - self._last_save >= self.interval_s):
            self.save(state_tree, meta)

    def finalize(self, training_state, meta: dict):
        """End of a finished run: keep networks + optimizer states in final_state.pkl (small) and remove
        the resume file latest.pkl, whose only extra content is the replay buffer and env state.
        Documented in PLAN.md / reports (a finished run is never resumed; DONE makes it skip)."""
        if not self.enabled:
            return
        final_path = os.path.join(self.run_dir, "final_state.pkl")
        host = jax.device_get(training_state)
        payload = {"meta": meta, "leaves": [np.asarray(x) for x in jax.tree_util.tree_leaves(host)]}
        tmp = final_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final_path)
        if os.path.exists(self.path):
            os.remove(self.path)
        logging.info("final state saved to %s; resume file removed", final_path)

    def load(self, template_tree):
        with open(self.path, "rb") as f:
            payload = pickle.load(f)
        t_leaves, treedef = jax.tree_util.tree_flatten(template_tree)
        leaves = payload["leaves"]
        if len(leaves) != len(t_leaves):
            raise ValueError(f"checkpoint has {len(leaves)} leaves, template has {len(t_leaves)}")
        for i, (a, b) in enumerate(zip(leaves, t_leaves)):
            if tuple(np.shape(a)) != tuple(np.shape(b)):
                raise ValueError(f"leaf {i}: checkpoint shape {np.shape(a)} != template shape {np.shape(b)}")
        restored = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in leaves])
        return restored, payload["meta"]


def write_marker(run_dir: Optional[str], name: str, content: dict):
    if not run_dir:
        return
    path = os.path.join(run_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(content, f, indent=2, default=str)
    os.replace(tmp, path)
