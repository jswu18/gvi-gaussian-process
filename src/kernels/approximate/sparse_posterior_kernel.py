from typing import Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import ApproximateBaseKernel
from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.utils.matrix_operations import add_diagonal_regulariser


class SparsePosteriorKernelParameters(KernelBaseParameters):
    """
    The parameters of the sparse posterior kernel, which are the parameters of the base kernel.
    """

    base_kernel: KernelBaseParameters


class SparsePosteriorKernel(ApproximateBaseKernel):
    """
    The sparse posterior kernel is a kernel that is used to approximate the posterior kernel of the
    regulariser Gaussian process.
    """

    Parameters = SparsePosteriorKernelParameters

    def __init__(
        self,
        base_kernel: KernelBase,
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        super().__init__(
            inducing_points=inducing_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> SparsePosteriorKernelParameters:
        return SparsePosteriorKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, SparsePosteriorKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        gram_inducing = self.base_kernel.calculate_gram(
            parameters=parameters.base_kernel,
            x1=self.inducing_points,
            x2=self.inducing_points,
        )
        gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=gram_inducing,
                diagonal_regularisation=self.diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=self.is_diagonal_regularisation_absolute_scale,
            )
        )
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
                c_and_lower=gram_inducing_cholesky_decomposition_and_lower,
                b=gram_x2_inducing.T,
            )
        )
