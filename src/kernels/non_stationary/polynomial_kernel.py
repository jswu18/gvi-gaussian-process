from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.non_stationary.base import (
    NonStationaryKernelBase,
    NonStationaryKernelBaseParameters,
)
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.utils.custom_types import JaxFloatType, PRNGKey


class PolynomialKernelParameters(NonStationaryKernelBaseParameters):
    """
    The parameters of a polynomial kernel.
    """

    log_constant: JaxFloatType
    log_scaling: JaxFloatType


class PolynomialKernel(NonStationaryKernelBase):
    """
    A polynomial kernel of the form (scaling * x1^T x2 + constant)^degree.
    """

    Parameters = PolynomialKernelParameters

    def __init__(
        self,
        polynomial_degree: float = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.polynomial_degree = polynomial_degree
        super().__init__(preprocess_function=preprocess_function)

    def _calculate_kernel(
        self,
        parameters: PolynomialKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.float64:
        return jnp.power(
            (
                jnp.exp(parameters.log_scaling) * jnp.dot(x1, x2.T)
                + jnp.exp(parameters.log_constant)
            ),
            self.polynomial_degree,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> PolynomialKernelParameters:
        return PolynomialKernel.Parameters(
            log_constant=parameters["log_constant"],
            log_scaling=parameters["log_scaling"],
        )
