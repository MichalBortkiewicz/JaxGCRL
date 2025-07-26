import logging
import math

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


EPS = 1e-8

def l2normalize(
    x: jnp.ndarray,
    axis: int,
) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    x = x / jnp.maximum(l2norm, EPS)

    return x

class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(1.0 * self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int

    def setup(self):
        self.w = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=False,  # important!
        )

    def __call__(self, x):
        return self.w(x)

class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = 1e-8

    def setup(self):
        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    def setup(self):
        self.w = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones((x.shape[:-1] + (1,))) * self.c_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)

        return x

class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float

    expansion: int = 4

    def setup(self):
        self.mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion),
        )
        self.alpha_scaler = Scaler(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        x = l2normalize(x, axis=-1)

        return x
    
class EncoderSimba(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False

    def setup(self):
        self.num_blocks: int = self.network_depth - 1  # Because we have also embedder
        self.hidden_dim: int = self.network_width
        self.scaler_init: float = math.sqrt(2 / self.network_width)
        self.scaler_scale: float = math.sqrt(2 / self.network_width)
        self.alpha_init: float = 1 / (self.num_blocks + 1)
        self.alpha_scale: float = 1 / math.sqrt(self.network_width)
        self.c_shift: float = 3

        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = data
        x = self.embedder(x)
        x = self.encoder(x)

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


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
