from typing import Any, Callable, Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary.base import (
    NonStationaryKernelBase,
    NonStationaryKernelBaseParameters,
)
from src.module import PYDANTIC_VALIDATION_CONFIG


class CustomMappingKernelParameters(KernelBaseParameters):
    """
    The parameters of a custom mapping kernel.
    """

    base_kernel: NonStationaryKernelBaseParameters
    feature_mapping: Any


class CustomMappingKernel(KernelBase):
    """
    A kernel that uses a custom function to map the inputs to a higher dimensional space
    and then applies a base kernel to the mapped inputs.
    """

    Parameters = CustomMappingKernelParameters

    def __init__(
        self,
        base_kernel: NonStationaryKernelBase,
        feature_mapping: Callable[[Any, jnp.ndarray], jnp.ndarray],
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        self.feature_mapping = feature_mapping
        KernelBase.__init__(self, preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> CustomMappingKernelParameters:
        return CustomMappingKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
            feature_mapping=parameters["feature_mapping"],
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, CustomMappingKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix with a custom mapping followed by a base kernel function.
            - m1 is the number of points in x1
            - m2 is the number of points in x2
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: design matrix of shape (m1, d)
            x2: design matrix of shape (m2, d)

        Returns: the kernel gram matrix of shape (m_1, m_2)

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return (
            jax.vmap(
                lambda x1_: jax.vmap(
                    lambda x2_: self.base_kernel.calculate_gram(
                        parameters=parameters.base_kernel,
                        x1=self.feature_mapping(
                            parameters.feature_mapping, x1_
                        ).reshape(1, -1),
                        x2=self.feature_mapping(
                            parameters.feature_mapping, x2_
                        ).reshape(1, -1),
                    )
                )(x2[:, None, ...])
            )(x1[:, None, ...])
        ).reshape(x1.shape[0], x2.shape[0])
