"""ScaledMLP: the residual + LayerNorm + Swish network of Wang et al. (2025), made configurable.

Copied in structure from scaling-crl@17acb51 train.py:108-158:
  stem:   Dense(width) -> [LN] -> act
  blocks: depth // 4 times { skip = x; 4 x (Dense(width) -> [LN] -> act); x = x + skip if residual }
  head:   Dense(out_size)   (no norm, no activation)
"Depth D" = the D Dense layers inside the blocks; the network has D + 2 Dense layers in total.
Every Dense layer uses variance_scaling(1/3, fan_in, uniform) weights and zero biases.
"""

from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

VARIANTS = {
    # name: (use_residual, use_ln, activation)
    "plain": (False, False, "relu"),
    "full": (True, True, "swish"),
    "full_no_ln": (True, False, "swish"),
    "full_no_res": (False, True, "swish"),
    "full_no_swish": (True, True, "relu"),
}


def variant_flags(name: str) -> dict:
    """Map a variant name to the network flags used by the agents."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown network variant {name!r}; known: {sorted(VARIANTS)}")
    use_residual, use_ln, activation = VARIANTS[name]
    return dict(use_residual=use_residual, use_ln=use_ln, activation=activation)


def get_activation(name: str):
    if name == "relu":
        return nn.relu
    if name == "swish":
        return nn.swish
    raise ValueError(f"Unknown activation {name!r}")


def check_depth(depth: int):
    if depth < 4 or depth % 4 != 0:
        raise ValueError(f"ScaledMLP depth must be a multiple of 4 and >= 4, got {depth}")


class ScaledTrunk(nn.Module):
    """Stem + residual blocks, without the output head."""

    width: int
    depth: int
    use_ln: bool
    use_residual: bool
    activation: Literal["relu", "swish"]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        check_depth(self.depth)
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        act = get_activation(self.activation)

        def layer(h):
            h = nn.Dense(self.width, kernel_init=init, bias_init=nn.initializers.zeros)(h)
            if self.use_ln:
                h = nn.LayerNorm()(h)
            h = act(h)
            self.sow("intermediates", "layer", h)  # no-op unless applied with mutable=["intermediates"]
            return h

        x = layer(x)  # stem
        for _ in range(self.depth // 4):
            skip = x
            for _ in range(4):
                x = layer(x)
            if self.use_residual:
                x = x + skip
        self.sow("intermediates", "trunk_out", x)
        return x


class ScaledMLP(nn.Module):
    """ScaledTrunk followed by a linear head of size out_size."""

    width: int
    depth: int
    out_size: int
    use_ln: bool
    use_residual: bool
    activation: Literal["relu", "swish"]
    head_ln: bool = False  # LayerNorm before the head (LAC critic, Kang et al. 2026, Eq. 9)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = ScaledTrunk(
            width=self.width,
            depth=self.depth,
            use_ln=self.use_ln,
            use_residual=self.use_residual,
            activation=self.activation,
        )(x)
        if self.head_ln:
            x = nn.LayerNorm()(x)
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        return nn.Dense(self.out_size, kernel_init=init, bias_init=nn.initializers.zeros)(x)
