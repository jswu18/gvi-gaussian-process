from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.module import Module


class MeanFunction(Module, ABC):
    @abstractmethod
    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        pass
