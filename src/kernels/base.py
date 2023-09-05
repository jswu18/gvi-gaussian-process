from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import pydantic

from src.module import PYDANTIC_VALIDATION_CONFIG, Module, ModuleParameters
from src.utils.checks import check_matching_dimensions, check_maximum_dimension


class KernelBaseParameters(ModuleParameters, ABC):
    """
    A base class for all kernel parameters. All kernel parameter classes will inheret this ABC.
    """

    pass


class KernelBase(Module, ABC):
    """
    A base class for all kernels. All kernel classes will inheret this ABC.
    """

    Parameters = KernelBaseParameters

    def __init__(
        self,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Construct for the KernelBase class.

        Args:
            number_output_dimensions: the number of output dimensions of the kernel function
            preprocess_function: a function to preprocess the inputs of the kernel function
        """
        self.number_output_dimensions = number_output_dimensions
        self._jit_compiled_calculate_gram = jax.jit(
            lambda parameters, x1, x2: self._calculate_gram(
                parameters=parameters, x1=x1, x2=x2
            )
        )
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def preprocess_inputs(
        self, x: jnp.ndarray, y: jnp.ndarray = None
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
        return (
            jnp.atleast_2d(self.preprocess_function(x)),
            jnp.atleast_2d(self.preprocess_function(y)),
        )

    @staticmethod
    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
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
        parameters: KernelBaseParameters,
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
        raise NotImplementedError

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_gram(
        self,
        parameters: KernelBaseParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray = None,
        full_covariance: bool = True,
    ) -> jnp.ndarray:
        """
        Computes the prior gram matrix of the kernel. If x2 is None, the covariance matrix is computed for x1 and x1.
            - m1 is the number of points in x1
            - m2 is the number of points in x2
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: design matrix of shape (m1, d)
            x2: design matrix of shape (m2, d)
            full_covariance: whether to compute the full covariance matrix or just the diagonal (requires m1 == m2)

        Returns: the kernel gram matrix of shape (m_1, m_2)
        """
        x1, x2 = self.preprocess_inputs(x1, x2)
        self.check_inputs(x1, x2)
        Module.check_parameters(parameters, self.Parameters)
        if full_covariance:
            return self._jit_compiled_calculate_gram(parameters.dict(), x1, x2)
        else:
            assert (
                x1.shape[0] == x2.shape[0]
            ), f"{x1.shape[0]=} must be equal to {x2.shape[0]=} for {full_covariance=}"
            return (
                jax.vmap(
                    lambda x1_, x2_: self._jit_compiled_calculate_gram(
                        parameters.dict(),
                        x1_,
                        x2_,
                    )
                )(x1[:, None, ...], x2[:, None, ...])
                .squeeze(axis=-1)
                .squeeze(axis=-1)
                .T
            )
