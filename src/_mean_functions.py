from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MeanFunction(ABC):
    @abstractmethod
    def predict(self, parameters: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        pass


class Constant(MeanFunction):
    def predict(self, x: jnp.ndarray, parameters: Dict[str, Any] = None) -> jnp.ndarray:
        if parameters is None:
            parameters = {"constant": 0}

        return parameters["constant"] * jnp.ones((x.shape[0],))


class MultiLayerPerceptron(MeanFunction, nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x.reshape(
            -1,
        )

    def predict(self, x: jnp.ndarray, parameters: Dict[str, Any]) -> jnp.ndarray:
        return self.apply(parameters, x)
