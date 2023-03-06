from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.module import Module


class Kernel(Module, ABC):
    @abstractmethod
    def calculate_gram(
        self, parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Computes the Gram matrix of the kernel. If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)

        """
        raise NotImplementedError
