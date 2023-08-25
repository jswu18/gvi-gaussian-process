from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import ApproximateBaseKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.matrix_operations import add_diagonal_regulariser


class FixedSparsePosteriorKernelParameters(KernelBaseParameters):
    base_kernel: KernelBaseParameters


class FixedSparsePosteriorKernel(ApproximateBaseKernel):
    Parameters = FixedSparsePosteriorKernelParameters

    def __init__(
        self,
        reference_kernel: KernelBase,
        reference_kernel_parameters: KernelBaseParameters,
        base_kernel: KernelBase,
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable = None,
    ):
        self.reference_kernel = reference_kernel
        self.reference_kernel_parameters = reference_kernel_parameters
        self.reference_gram_inducing_cholesky = jnp.linalg.inv(
            jnp.linalg.cholesky(
                self.reference_kernel.calculate_gram(
                    parameters=reference_kernel_parameters,
                    x1=inducing_points,
                    x2=inducing_points,
                )
            )
        )
        self.reference_gram_inducing_inverse = jnp.dot(
            self.reference_gram_inducing_cholesky,
            self.reference_gram_inducing_cholesky.T,
        )
        self.base_kernel = base_kernel
        super().__init__(
            inducing_points=inducing_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> FixedSparsePosteriorKernelParameters:
        return FixedSparsePosteriorKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, FixedSparsePosteriorKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        gram_x1_inducing = self.base_kernel.calculate_gram(
            parameters=parameters.base_kernel,
            x1=x1,
            x2=self.inducing_points,
        )
        gram_x2_inducing = self.base_kernel.calculate_gram(
            parameters=parameters.base_kernel,
            x1=x2,
            x2=self.inducing_points,
        )
        gram_x1_x2 = self.base_kernel.calculate_gram(
            parameters=parameters.base_kernel,
            x1=x1,
            x2=x2,
        )
        return gram_x1_x2 - (
            gram_x1_inducing
            @ cho_solve(
                c_and_lower=self.reference_gram_inducing_cholesky_decomposition_and_lower,
                b=gram_x2_inducing.T,
            )
        )
