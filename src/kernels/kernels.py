from abc import ABC, abstractmethod

import jax.numpy as jnp
import pydantic

from src.module import Module
from src.parameters.kernels.kernels import KernelParameters


class Kernel(Module, ABC):
    Parameters = KernelParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def calculate_gram(
        self,
        parameters: KernelParameters,
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
