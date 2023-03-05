from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.module import Module


class Kernel(Module, ABC):
    @abstractmethod
    def gram(
        self, parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        raise NotImplementedError
