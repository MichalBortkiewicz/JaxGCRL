import jax
import jax.numpy as jnp

def mrn_distance(x, y):
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = jnp.max(jax.nn.relu(x_prefix - y_prefix), axis=-1)
    l2_component = jnp.linalg.norm(x_suffix - y_suffix, axis=-1)
    return max_component + l2_component
