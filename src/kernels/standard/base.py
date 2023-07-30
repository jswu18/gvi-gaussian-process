from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
import pydantic
from jax import vmap

from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import Module


class StandardKernelBaseParameters(KernelBaseParameters, ABC):
    pass


class StandardKernelBase(KernelBase, ABC):
    """
    Kernels that can be easily defined with a kernel function evaluated at a single pair of points.
    """

    def __init__(
        self, preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None
    ):
        super().__init__(preprocess_function=preprocess_function)

    @staticmethod
    @abstractmethod
    def _calculate_kernel(
        parameters: StandardKernelBaseParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.float64:
        """
        Computes the kernel function for a single pair of points x and y.
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x: vector of shape (1, d)
            y: vector of shape (1, d)

        Returns: the kernel function evaluated at x and y
        """
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_kernel(
        self,
        parameters: StandardKernelBaseParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray = None,
    ) -> jnp.float64:
        """
        Computes the kernel function for a single pair of points x and y. Calls
        _calculate_kernel after preprocessing the inputs, which needs to be
        implemented by the child class. If y is None, the kernel function is
         computed for x and x.
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: vector of shape (1, d)
            x2: vector of shape (1, d) or None

        Returns: the kernel function evaluated at x and y
        """
        x1, x2 = self.preprocess_inputs(x1, x2)
        self.check_inputs(x1, x2)
        Module.check_parameters(parameters, self.Parameters)
        return self._calculate_kernel(parameters, x1, x2)

    def _calculate_gram(
        self,
        parameters: StandardKernelBaseParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the prior gram matrix of the kernel.
            - m1 is the number of points in x1
            - m2 is the number of points in x2
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: design matrix of shape (m1, d)
            x2: design matrix of shape (m2, d)

        Returns: the kernel gram matrix of shape (m_1, m_2)
        """
        return vmap(
            lambda x1_: vmap(
                lambda x2_: self.calculate_kernel(
                    parameters=parameters,
                    x1=x1_,
                    x2=x2_,
                )
            )(x2[:, None, ...])
        )(x1[:, None, ...])
