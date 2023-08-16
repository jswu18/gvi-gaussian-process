from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.custom_kernel import CustomKernel, CustomKernelParameters
from src.utils.matrix_operations import add_diagonal_regulariser


class CustomApproximateKernelParameters(
    ApproximateBaseKernelParameters, CustomKernelParameters
):
    pass


class CustomApproximateKernel(CustomKernel, ApproximateBaseKernel):
    """
    Approximate kernels which are defined with respect to a reference kernel
    """

    Parameters = CustomApproximateKernelParameters

    def __init__(
        self,
        kernel_function: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.float64],
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable = None,
    ):
        self.kernel_function = kernel_function
        CustomKernel.__init__(
            self,
            kernel_function=kernel_function,
            preprocess_function=preprocess_function,
        )
        ApproximateBaseKernel.__init__(
            self,
            inducing_points=inducing_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, CustomKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        custom_mapping_gram_inducing = CustomKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=self.inducing_points,
            x2=self.inducing_points,
        )
        custom_mapping_gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=custom_mapping_gram_inducing,
                diagonal_regularisation=self.diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=self.is_diagonal_regularisation_absolute_scale,
            )
        )
        custom_mapping_gram_x1_inducing = CustomKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=x1,
            x2=self.inducing_points,
        )
        custom_mapping_gram_x2_inducing = CustomKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=x2,
            x2=self.inducing_points,
        )
        custom_mapping_gram_x1_x2 = CustomKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=x1,
            x2=x2,
        )
        return custom_mapping_gram_x1_x2 - (
            custom_mapping_gram_x1_inducing
            @ cho_solve(
                c_and_lower=custom_mapping_gram_inducing_cholesky_decomposition_and_lower,
                b=custom_mapping_gram_x2_inducing.T,
            )
        )
