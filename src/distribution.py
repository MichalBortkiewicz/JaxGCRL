# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probability distributions in JAX."""

import abc
import jax
import jax.numpy as jnp


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, event_ndims, reparametrizable):
        """Abstract class for parametric (action) distribution.

        Specifies how to transform distribution parameters (i.e. actor output)
        into a distribution over actions.

        Args:
          param_size: size of the parameters for the distribution
          event_ndims: rank of the distribution sample (i.e. action)
          reparametrizable: is the distribution reparametrizable
        """
        self._param_size = param_size
        self._event_ndims = event_ndims  # rank of events
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        """Creates distribution from parameters."""
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return jnp.tanh(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    # OK
    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    # OK
    def mode(self, parameters):
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, x_ts):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=dist.loc, scale=dist.scale)
        log_prob -= jnp.log((1 - jnp.square(jnp.tanh(x_ts))) + 1e-6)
        log_prob = log_prob.sum(-1)
        return log_prob


class NormalDistribution:
    """Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape, dtype=self.loc.dtype) * self.scale + self.loc

    def mode(self):
        return self.loc


class NormalTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, log_min_std=-5, log_max_std=2):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.
          var_scale: adjust the gaussian's scale parameter.
        """

        super().__init__(param_size=2 * event_size, event_ndims=1, reparametrizable=True)
        self.log_max_std = log_max_std
        self.log_min_std = log_min_std

    def create_dist(self, parameters):
        loc, log_std = jnp.split(parameters, 2, axis=-1)
        log_std = jnp.tanh(log_std)
        log_std = self.log_min_std + 0.5 * (self.log_max_std - self.log_min_std) * (log_std + 1)
        scale = jnp.exp(log_std)
        return NormalDistribution(loc=loc, scale=scale)
