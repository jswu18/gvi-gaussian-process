from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.non_stationary.base import (
    NonStationaryKernelBase,
    NonStationaryKernelBaseParameters,
)
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.utils.custom_types import JaxFloatType


class InnerProductKernelParameters(NonStationaryKernelBaseParameters):
    """
    The parameters of an inner product kernel.
    """

    log_scaling: JaxFloatType


class InnerProductKernel(NonStationaryKernelBase):
    """
    An inner product kernel of the form scaling * x1^T x2.
    """

    Parameters = InnerProductKernelParameters

    def __init__(
        self,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        super().__init__(preprocess_function=preprocess_function)

    def _calculate_kernel(
        self,
        parameters: InnerProductKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.float64:
        return jnp.exp(parameters.log_scaling) * jnp.dot(x1, x2.T)

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> InnerProductKernelParameters:
        return InnerProductKernel.Parameters(
            log_scaling=parameters["log_scaling"],
        )
