from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.non_stationary.base import (
    NonStationaryKernelBase,
    NonStationaryKernelBaseParameters,
)
from src.utils.custom_types import JaxFloatType


class InnerProductKernelParameters(NonStationaryKernelBaseParameters):
    log_scaling: JaxFloatType


class InnerProductKernel(NonStationaryKernelBase):
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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> InnerProductKernelParameters:
        return InnerProductKernel.Parameters(
            log_scaling=parameters["log_scaling"],
        )
