from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


class MeanFunction(ABC):
    @abstractmethod
    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        pass
