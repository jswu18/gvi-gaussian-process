from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
import pydantic

from src.module import Module
from src.parameters.kernels.kernels import KernelParameters
from src.utils.checks import check_matching_dimensions, check_maximum_dimension


class Kernel(Module, ABC):
    Parameters = KernelParameters

    @staticmethod
    def preprocess_inputs(
        x: jnp.ndarray, y: jnp.ndarray = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Preprocesses the inputs of the kernel function.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)
        """
        y = x if y is None else y
        return jnp.atleast_2d(x), jnp.atleast_2d(y)

    @staticmethod
    def check_inputs(x: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Checks the inputs of a kernel function.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)
        """
        check_matching_dimensions(x, y)
        check_maximum_dimension(x, maximum_dimensionality=2)
        check_maximum_dimension(y, maximum_dimensionality=2)

    @abstractmethod
    def _calculate_gram(
        self,
        parameters: KernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_gram(
        self,
        parameters: KernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Computes the Gram matrix of the kernel.
        Calls _calculate_gram, which needs to be implemented by the child class.
        If y is None, the gram is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d) if y is None, compute for x and x

        Returns: the kernel gram matrix of shape (n, m)

        """
        x, y = self.preprocess_inputs(x, y)
        self.check_inputs(x, y)
        return self._calculate_gram(parameters, x, y)
