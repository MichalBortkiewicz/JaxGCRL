import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


class ScaledEncoder(nn.Module):
    """network_type="scaled": ScaledMLP trunk (stem + depth/4 residual blocks, LN, Swish) + Dense(repr_dim)."""

    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    use_ln: bool = True
    use_residual: bool = True
    activation: str = "swish"

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        from jaxgcrl.agents.common.scaled_net import ScaledMLP

        return ScaledMLP(
            width=self.network_width,
            depth=self.network_depth,
            out_size=self.repr_dim,
            use_ln=self.use_ln,
            use_residual=self.use_residual,
            activation=self.activation,
        )(data)


class ScaledActor(nn.Module):
    """network_type="scaled": ScaledTrunk + the original mean / log-std heads of Actor above."""

    action_size: int
    network_width: int = 256
    network_depth: int = 4
    use_ln: bool = True
    use_residual: bool = True
    activation: str = "swish"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        from jaxgcrl.agents.common.scaled_net import ScaledTrunk

        x = ScaledTrunk(
            width=self.network_width,
            depth=self.network_depth,
            use_ln=self.use_ln,
            use_residual=self.use_residual,
            activation=self.activation,
        )(x)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
