from typing import Any, Callable, Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.kernels.non_stationary.base import (
    NonStationaryKernelBase,
    NonStationaryKernelBaseParameters,
)
from src.utils.custom_types import PRNGKey


class NeuralNetworkKernelParameters(KernelBaseParameters):
    base_kernel: NonStationaryKernelBaseParameters
    neural_network: Any


class NeuralNetworkKernel(KernelBase):
    """
    A wrapper class for the kernel function provided by the NTK package.
    """

    Parameters = NeuralNetworkKernelParameters

    def __init__(
        self,
        base_kernel: NonStationaryKernelBase,
        neural_network: nn.Module,
        preprocess_function: Callable = None,
    ):
        self.base_kernel = base_kernel
        self.neural_network = neural_network
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NeuralNetworkKernelParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        Args:
            parameters: A dictionary of the parameters for Neural Network Gaussian Process Kernel.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernel.

        """
        return NeuralNetworkKernel.Parameters(
            base_kernel=self.base_kernel.generate_parameters(parameters["base_kernel"]),
            neural_network=parameters["neural_network"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NeuralNetworkKernelParameters:
        """
        Initialise the parameters of the Neural Network Gaussian Process Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Gaussian Process Kernels.

        """
        pass

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, NeuralNetworkKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return (
            jax.vmap(
                lambda x1_: jax.vmap(
                    lambda x2_: self.base_kernel.calculate_gram(
                        parameters=parameters.base_kernel,
                        x1=self.neural_network.apply(
                            parameters.neural_network, x1_
                        ).reshape(1, -1),
                        x2=self.neural_network.apply(
                            parameters.neural_network, x2_
                        ).reshape(1, -1),
                    )
                )(x2[:, None, ...])
            )(x1[:, None, ...])
        ).reshape(x1.shape[0], x2.shape[0])
