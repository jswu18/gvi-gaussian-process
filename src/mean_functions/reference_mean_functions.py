from typing import Any

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.mean_functions.mean_functions import MeanFunction

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantFunction(MeanFunction):
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return FrozenDict({"constant": kwargs["constant"]})

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return parameters["constant"] * jnp.ones((x.shape[0],))
