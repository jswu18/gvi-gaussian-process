from abc import ABC, abstractmethod
from typing import Dict

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src import decorators
from src.module import Module


class Kernel(Module, ABC):
    parameter_keys: Dict[str, type] = NotImplementedError

    @decorators.common.check_parameters(parameter_keys)
    @abstractmethod
    def calculate_gram(
        self,
        parameters: FrozenDict,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Computes the Gram matrix of the kernel.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d) if y is None, compute for x and x

        Returns: the kernel gram matrix of shape (n, m)

        """
        raise NotImplementedError
