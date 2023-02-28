import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.mean_functions.mean_functions import MeanFunction


class ConstantFunction(MeanFunction):
    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return parameters["constant"] * jnp.ones((x.shape[0],))
