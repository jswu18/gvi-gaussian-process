from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.custom_mapping_kernel import (
    CustomMappingKernel,
    CustomMappingKernelParameters,
)
from src.kernels.non_stationary.base import NonStationaryKernelBase
from src.utils.matrix_operations import add_diagonal_regulariser


class CustomMappingApproximateKernelParameters(
    ApproximateBaseKernelParameters, CustomMappingKernelParameters
):
    pass


class CustomMappingApproximateKernel(CustomMappingKernel, ApproximateBaseKernel):
    """
    Approximate kernels which are defined with respect to a reference kernel
    """

    Parameters = CustomMappingApproximateKernelParameters

    def __init__(
        self,
        base_kernel: NonStationaryKernelBase,
        feature_mapping: Callable[[Any, jnp.ndarray], jnp.ndarray],
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        self.feature_mapping = feature_mapping
        self.diagonal_regularisation = diagonal_regularisation
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        CustomMappingKernel.__init__(
            self,
            base_kernel=base_kernel,
            feature_mapping=feature_mapping,
            preprocess_function=preprocess_function,
        )
        ApproximateBaseKernel.__init__(
            self,
            inducing_points=inducing_points,
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, CustomMappingKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        custom_mapping_gram_inducing = CustomMappingKernel._calculate_gram(
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
        custom_mapping_gram_x1_inducing = CustomMappingKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=x1,
            x2=self.inducing_points,
        )
        custom_mapping_gram_x2_inducing = CustomMappingKernel._calculate_gram(
            self,
            parameters=parameters,
            x1=x2,
            x2=self.inducing_points,
        )
        custom_mapping_gram_x1_x2 = CustomMappingKernel._calculate_gram(
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
