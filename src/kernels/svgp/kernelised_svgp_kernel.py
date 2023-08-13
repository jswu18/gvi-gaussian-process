from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_solve

from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.svgp.base import SVGPBaseKernel, SVGPBaseKernelParameters


class KernelisedSVGPKernelParameters(SVGPBaseKernelParameters):
    base_kernel: KernelBaseParameters


class KernelisedSVGPKernel(SVGPBaseKernel):
    Parameters = KernelisedSVGPKernelParameters

    def __init__(
        self,
        base_kernel: KernelBase,
        reference_kernel: KernelBase,
        reference_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.base_kernel = base_kernel
        super().__init__(
            reference_kernel_parameters=reference_kernel_parameters,
            reference_kernel=reference_kernel,
            preprocess_function=preprocess_function,
            log_observation_noise=log_observation_noise,
            inducing_points=inducing_points,
            training_points=training_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> KernelisedSVGPKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
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
        reference_gram_x1_inducing = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x1,
            x2=self.inducing_points,
        )

        reference_gram_x2_inducing = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x2,
            x2=self.inducing_points,
        )

        reference_gram_x1_x2 = self.reference_kernel.calculate_gram(
            parameters=self.reference_kernel_parameters,
            x1=x1,
            x2=x2,
        )

        return (
            reference_gram_x1_x2
            - (
                reference_gram_x1_inducing
                @ cho_solve(
                    c_and_lower=self.reference_gram_inducing_cholesky_decomposition_and_lower,
                    b=reference_gram_x2_inducing.T,
                )
            )
            + self.base_kernel.calculate_gram(
                parameters=parameters.base_kernel, x1=x1, x2=x2
            )
        )
