from typing import Any, Callable, Dict, Literal, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType

PRNGKey = Any  # pylint: disable=invalid-name


class TemperedKernelParameters(KernelBaseParameters):
    log_tempering_factor: Union[JaxArrayType[Literal["float64"]], JaxFloatType]


class TemperedKernel(KernelBase):

    Parameters = TemperedKernelParameters

    def __init__(
        self,
        base_kernel: KernelBase,
        base_kernel_parameters: KernelBaseParameters,
        number_output_dimensions: int,
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        self.base_kernel_parameters = base_kernel_parameters
        super().__init__(
            preprocess_function=preprocess_function,
            number_output_dimensions=number_output_dimensions,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> TemperedKernelParameters:
        return TemperedKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> TemperedKernelParameters:
        pass

    def _calculate_gram(
        self,
        parameters: TemperedKernelParameters,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        gram = self.base_kernel.calculate_gram(
            self.base_kernel_parameters,
            x1,
            x2,
        )
        tempered_gram = jnp.multiply(
            jnp.atleast_1d(jnp.exp(parameters.log_tempering_factor))[:, None, None],
            jnp.atleast_3d(gram),
        )
        return tempered_gram.reshape(gram.shape)
