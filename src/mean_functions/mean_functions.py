from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.module import Module


class MeanFunction(Module, ABC):
    @abstractmethod
    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
            - n is the number of points in x
            - d is the number of dimensions
        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, 1)

        """
        pass
