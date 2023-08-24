from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.custom_kernel import CustomKernel
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
    Parameters = CustomMappingApproximateKernelParameters

    def __init__(
        self,
        base_kernel: Union[NonStationaryKernelBase, CustomKernel, CustomMappingKernel],
        feature_mapping: Callable[[Any, jnp.ndarray], jnp.ndarray],
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        self.feature_mapping = feature_mapping
        CustomMappingKernel.__init__(
            self,
            base_kernel=base_kernel,
            feature_mapping=feature_mapping,
            preprocess_function=preprocess_function,
        )
        ApproximateBaseKernel.__init__(
            self,
            inducing_points=inducing_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> CustomMappingApproximateKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
        return CustomMappingApproximateKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
            feature_mapping=parameters["feature_mapping"],
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
