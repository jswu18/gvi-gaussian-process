from typing import Callable, Dict, Literal, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.standard.base import StandardKernelBase, StandardKernelBaseParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType, PRNGKey


class ARDKernelParameters(StandardKernelBaseParameters):
    log_scaling: JaxFloatType
    log_lengthscales: JaxArrayType[Literal["float64"]]


class ARDKernel(StandardKernelBase):
    Parameters = ARDKernelParameters

    def __init__(
        self,
        number_of_dimensions: int,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.number_of_dimensions = number_of_dimensions
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ARDKernelParameters:
        """
        Generates a Pydantic model of the parameters for ARDKernel Kernels.

        Args:
            parameters: A dictionary of the parameters for ARDKernel Kernels.

        Returns: A Pydantic model of the parameters for ARDKernel Kernels.

        """
        return ARDKernel.Parameters(
            log_scaling=parameters["log_scaling"],
            log_lengthscales=parameters["log_lengthscales"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ARDKernelParameters:
        """
        Initialise the parameters of the ARDKernel Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for ARDKernel Kernels.

        """
        pass

    @staticmethod
    def _calculate_kernel(
        parameters: ARDKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.float64:
        """
        The ARDKernel kernel function defined as:
        k(x1, x2) = scaling * exp(-0.5 * (x1 - x2)^T @ diag(1 / lengthscales) @ (x1 - x2)).
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: vector of shape (1, d)
            x2: vector of shape (1, d)

        Returns: the ARDKernel kernel function evaluated at x and y

        """
        scaling = jnp.exp(jnp.atleast_1d(parameters.log_scaling)) ** 2
        return jnp.sum(
            scaling
            * jnp.exp(
                -0.5
                * jnp.atleast_1d(jnp.exp(parameters.log_lengthscales))
                @ jnp.square(x1 - x2).T
            )
        ).astype(jnp.float64)
