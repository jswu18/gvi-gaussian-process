from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_solve

from src.kernels.approximate.svgp.base import SVGPBaseKernel, SVGPBaseKernelParameters
from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG


class KernelisedSVGPKernelParameters(SVGPBaseKernelParameters):
    """
    The parameters of the base kernel used to parameterise the SVGP kernel.
    """

    base_kernel: KernelBaseParameters


class KernelisedSVGPKernel(SVGPBaseKernel):
    """
    Parameterises the SVGP kernel using a base kernel.
    """

    Parameters = KernelisedSVGPKernelParameters

    def __init__(
        self,
        base_kernel: KernelBase,
        regulariser_kernel: KernelBase,
        regulariser_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.base_kernel = base_kernel
        super().__init__(
            regulariser_kernel_parameters=regulariser_kernel_parameters,
            regulariser_kernel=regulariser_kernel,
            preprocess_function=preprocess_function,
            log_observation_noise=log_observation_noise,
            inducing_points=inducing_points,
            training_points=training_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> KernelisedSVGPKernelParameters:
        return KernelisedSVGPKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, KernelisedSVGPKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        regulariser_gram_x1_inducing = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x1,
            x2=self.inducing_points,
        )

        regulariser_gram_x2_inducing = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x2,
            x2=self.inducing_points,
        )

        regulariser_gram_x1_x2 = self.regulariser_kernel.calculate_gram(
            parameters=self.regulariser_kernel_parameters,
            x1=x1,
            x2=x2,
        )

        return (
            regulariser_gram_x1_x2
            - (
                regulariser_gram_x1_inducing
                @ cho_solve(
                    c_and_lower=self.regulariser_gram_inducing_cholesky_decomposition_and_lower,
                    b=regulariser_gram_x2_inducing.T,
                )
            )
            + self.base_kernel.calculate_gram(
                parameters=parameters.base_kernel, x1=x1, x2=x2
            )
        )
