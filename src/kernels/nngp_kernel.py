from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters

PRNGKey = Any  # pylint: disable=invalid-name


class NNGPKernelParameters(KernelBaseParameters):
    pass


class NNGPKernel(KernelBase):
    """
    A wrapper class for the kernel function provided by the NTK package.
    """

    Parameters = NNGPKernelParameters

    def __init__(self, kernel_function: Callable, preprocess_function: Callable = None):
        """
        Define a Neural Network Gaussian Process Kernel using the kernel function provided by the NTK package.

        Args:
            kernel_function: The kernel function provided by the NTK package.
            preprocess_function: preprocess inputs before passing to kernel function
        """
        self.kernel_function = kernel_function
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NNGPKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
        return NNGPKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NNGPKernelParameters:
        """
        Initialise the parameters of the Neural Network Gaussian Process Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernels.

        """
        pass

    def _calculate_gram(
        self,
        parameters: NNGPKernelParameters,
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
        return self.kernel_function(
            x1,
            x2,
            "nngp",
        )