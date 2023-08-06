from typing import Any, Callable, Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


class CustomKernelParameters(KernelBaseParameters):
    custom: Any


class CustomKernel(KernelBase):
    """
    A wrapper class for the kernel function provided by the NTK package.
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
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> CustomKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
        return CustomKernel.Parameters(
            custom=parameters["custom"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> CustomKernelParameters:
        """
        Initialise the parameters of the Neural Network Gaussian Process Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernels.

        """
        pass

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, CustomKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix using the NNGP kernel function. If y is None, the Gram matrix is computed for x and x.
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
