from typing import Callable, Dict, Literal, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType, PRNGKey


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
        return TemperedKernel.Parameters(
            log_tempering_factor=parameters["log_tempering_factor"]
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, TemperedKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
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
