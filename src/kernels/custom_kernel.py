from typing import Any, Callable, Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG


class CustomKernelParameters(KernelBaseParameters):
    """
    The parameters of a custom kernel where the parameters can be any type.
    """

    custom: Any


class CustomKernel(KernelBase):
    """
    A wrapper class for any custom kernel function.
    """

    Parameters = CustomKernelParameters

    def __init__(
        self,
        kernel_function: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.float64],
        preprocess_function: Callable = None,
    ):
        """
        Define a kernel using a custom kernel function.

        Args:
            kernel_function: The kernel function provided by the NTK package.
            preprocess_function: preprocess inputs before passing to kernel function
        """
        self.kernel_function = kernel_function
        KernelBase.__init__(self, preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> CustomKernelParameters:
        return CustomKernel.Parameters(
            custom=parameters["custom"],
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, CustomKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix with a custom kernel function.
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
                    lambda x2_: self.kernel_function(parameters.custom, x1_, x2_)
                )(x2[:, None, ...])
            )(x1[:, None, ...])
        ).reshape(x1.shape[0], x2.shape[0])
