import jax
from jax import numpy as jnp


def sample_circle(radius, rng) -> jax.Array:
    ang = jnp.pi * 2.0 * jax.random.uniform(rng)
    target_x = radius * jnp.cos(ang)
    target_y = radius * jnp.sin(ang)
    return jax.Array([target_x, target_y])

# NOTE: This is NOT uniform sampling from a disk. 
# Here both angle and distance are sampled uniformly
def sample_disk(rng, min_radius=0.0, max_radius=1.0) -> jax.Array:
    rng, rng1 = jax.random.split(rng, 2)
    radius = jax.random.uniform(rng, minval=min_radius, maxval=max_radius)
    ang = jnp.pi * 2.0 * jax.random.uniform(rng1)
    target_x = radius * jnp.cos(ang)
    target_y = radius * jnp.sin(ang)
    return jnp.array([target_x, target_y])



