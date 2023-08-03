from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


class MultiOutputKernelParameters(KernelBaseParameters):
    kernels: List[KernelBaseParameters]


class MultiOutputKernel(KernelBase):
    Parameters = MultiOutputKernelParameters

    def __init__(self, kernels: List[KernelBase]):
        assert all(kernel.number_output_dimensions == 1 for kernel in kernels)
        self.kernels = kernels
        super().__init__(
            number_output_dimensions=len(kernels),
            preprocess_function=None,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> MultiOutputKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
        assert len(parameters["kernels"]) == self.number_output_dimensions
        return MultiOutputKernel.Parameters(
            kernels=[
                kernel.generate_parameters(parameters=parameters_)
                for kernel, parameters_ in zip(self.kernels, parameters["kernels"])
            ]
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MultiOutputKernelParameters:
        """
        Initialise the parameters of the Neural Network Gaussian Process Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernels.

        """
        kernel_parameters = []
        for kernel in self.kernels:
            key, subkey = jax.random.split(key=key)
            kernel_parameters.append(kernel.initialise_random_parameters(key=subkey))

        return MultiOutputKernel.Parameters(kernels=kernel_parameters)

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, MultiOutputKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the prior gram matrix of multiple kernels.
            - k is the number of kernels
            - m1 is the number of points in x1
            - m2 is the number of points in x2
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: design matrix of shape (m1, d)
            x2: design matrix of shape (m2, d)

        Returns: the kernel a stacked gram matrix of shape (k, m_1, m_2)
        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return jnp.array(
            [
                kernel_.calculate_gram(
                    parameters=parameters_,
                    x1=x1,
                    x2=x2,
                )
                for kernel_, parameters_ in zip(self.kernels, parameters.kernels)
            ]
        )
