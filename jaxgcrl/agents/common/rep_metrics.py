"""Representation diagnostics: weight norms, feature rank, dormant-neuron fraction.

Computed once per epoch (not per update) on `rep_batch` states sampled from the replay buffer, so they cost
one forward pass and one small SVD per network per epoch. They never touch the training RNG stream or the
training state.

Definitions (all on the scaled networks; the legacy MLP only gets weight norms):
- weight norms: total L2 norm over all Dense kernels of a network (`*_wnorm`), and the largest per-layer
  Frobenius norm (`*_wnorm_max`). Biases and LayerNorm scales are excluded.
- feature rank on the trunk output phi (B x width, before the linear head), from its singular values s:
    srank  (Kumar et al. 2021):  min k with sum_{i<=k} s_i >= (1 - 0.01) * sum_i s_i
    erank  (Roy & Vetterli 2007): exp(entropy(s / sum s))
- dormant fraction (Sokar et al. 2023): neuron i of a hidden layer is dormant if
    E|h_i| / mean_j E|h_j| <= tau,  tau = 0.1 (`*_dormant`), on post-activation outputs, averaged over all
    hidden layers; `*_dormant_last` is the same for the trunk output only.
- feature norm: mean L2 norm of phi (`*_fnorm`).
For the twin critic the values are averaged over the two Q-networks.
"""

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

DORMANT_TAU = 0.1
SRANK_DELTA = 0.01


def weight_norms(params):
    flat = flatten_dict(params)
    sq = [jnp.sum(jnp.square(v)) for k, v in flat.items() if k[-1] == "kernel"]
    sq = jnp.stack(sq)
    return jnp.sqrt(jnp.sum(sq)), jnp.sqrt(jnp.max(sq))


def srank(s, delta=SRANK_DELTA):
    c = jnp.cumsum(s) / jnp.maximum(jnp.sum(s), 1e-12)
    return jnp.sum(c < 1.0 - delta) + 1


def erank(s):
    p = s / jnp.maximum(jnp.sum(s), 1e-12)
    return jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-30)), 0.0)))


def dormant_fraction(h, tau=DORMANT_TAU):
    a = jnp.mean(jnp.abs(h), axis=0)
    score = a / jnp.maximum(jnp.mean(a), 1e-12)
    return jnp.mean(score <= tau)


def feature_stats(layers, trunk_out):
    """layers: list of (B, width) hidden outputs; trunk_out: (B, width)."""
    s = jnp.linalg.svd(trunk_out, compute_uv=False)
    return {
        "srank": srank(s).astype(jnp.float32),
        "erank": erank(s),
        "dormant": jnp.mean(jnp.stack([dormant_fraction(h) for h in layers])),
        "dormant_last": dormant_fraction(trunk_out),
        "fnorm": jnp.mean(jnp.linalg.norm(trunk_out, axis=-1)),
    }


def _sown(state, name):
    """All values sown under `name`, grouped per ScaledTrunk (in tree order). Returns a list of lists."""
    groups = []
    flat = flatten_dict(state["intermediates"])
    for k in sorted(flat):
        if k[-1] == name:
            groups.append(list(flat[k]))
    return groups


def network_rep_metrics(prefix, params, features_fn, *inputs):
    """features_fn(params, *inputs) -> flax mutable state with sown 'layer' and 'trunk_out' intermediates,
    or None for networks without feature hooks (then only weight norms)."""
    wn, wmax = weight_norms(params)
    out = {f"{prefix}_wnorm": wn, f"{prefix}_wnorm_max": wmax}
    if features_fn is None:
        return out
    state = features_fn(params, *inputs)
    layer_groups, trunk_groups = _sown(state, "layer"), _sown(state, "trunk_out")
    stats = [feature_stats(l, t[0]) for l, t in zip(layer_groups, trunk_groups)]
    for key in stats[0]:
        out[f"{prefix}_{key}"] = jnp.mean(jnp.stack([s[key] for s in stats]))
    return out


def sample_states(replay_buffer, buffer_state, key, n):
    """n random (observation, action) pairs from the replay buffer. The buffer state returned by sample()
    is discarded, so the training sampling stream is unchanged."""
    _, tr = replay_buffer.sample(buffer_state)
    obs = jnp.reshape(tr.observation, (-1, tr.observation.shape[-1]))
    act = jnp.reshape(tr.action, (-1, tr.action.shape[-1]))
    idx = jax.random.randint(key, (n,), 0, obs.shape[0])
    return obs[idx], act[idx]
