from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import vmap

from src.kernels.kernels import Kernel
from src.parameters.kernels.reference_kernels import (
    ARDKernelParameters,
    NeuralNetworkGaussianProcessKernelParameters,
    StandardKernelParameters,
)
from src.utils import decorators

PRNGKey = Any  # pylint: disable=invalid-name


class StandardKernel(Kernel, ABC):
    Parameters = StandardKernelParameters
    """
    Kernel class for kernels that can be easily defined with a kernel function evaluated at a single pair of points.
    """

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def calculate_kernel(
        parameters: StandardKernelParameters, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.float64:
        """
        Computes the kernel function for a single pair of points x and y.
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (1, d)
            y: design matrix of shape (1, d)

        Returns: the kernel function evaluated at x and y
        """
        raise NotImplementedError


class ARDKernel(StandardKernel):
    Parameters = ARDKernelParameters

    def __init__(self, number_of_dimensions: int):
        self.number_of_dimensions = number_of_dimensions

    # def generate_parameters(self, parameters: FrozenDict) -> ARDKernelParameters:
    #     return self._generate_parameters(parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ARDKernelParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return ARDKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ARDKernelParameters:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    @decorators.preprocess_inputs
    @decorators.check_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_kernel(
        self, parameters: ARDKernelParameters, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        The ARD kernel function defined as:
        k(x, y) = scaling * exp(-0.5 * (x - y)^T @ diag(1 / lengthscales) @ (x - y)).
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        The parameters of the kernel are passed in as a dictionary of the form:
            parameters = {
                "log_scaling": float,
                "log_lengthscales": jnp.ndarray, of shape (d, 1),
            }

        Args:
            parameters: parameters of the kernel
            x: vector of shape (1, d)
            y: vector of shape (1, d) if y is None, compute for x and x

        Returns: the ARD kernel function evaluated at x and y

        """
        # if y is None, compute for x and x
        if y is None:
            y = x

        scaling = jnp.exp(jnp.atleast_1d(parameters.log_scaling)) ** 2
        lengthscale_matrix = jnp.diag(
            jnp.exp(jnp.atleast_1d(parameters.log_lengthscales))
        )
        return jnp.sum(
            scaling * jnp.exp(-0.5 * (x - y) @ lengthscale_matrix @ (x - y).T)
        ).astype(jnp.float64)

    @decorators.preprocess_inputs
    @decorators.check_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_gram(
        self,
        parameters: ARDKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Computes the Gram matrix of the kernel. If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)
        """
        # if y is None, compute for x and x
        if y is None:
            y = x

        return vmap(
            lambda x_: vmap(
                lambda y_: self.calculate_kernel(
                    parameters=parameters,
                    x=x_,
                    y=y_,
                )
            )(y)
        )(x)


class NeuralNetworkGaussianProcessKernel(Kernel):
    Parameters = NeuralNetworkGaussianProcessKernelParameters

    def __init__(self, ntk_kernel_function: Callable):
        """
        A wrapper class for the kernel function provided by the NTK package.
        Args:
            ntk_kernel_function: The kernel function provided by the NTK package.
        """
        self.ntk_kernel_function = ntk_kernel_function

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NeuralNetworkGaussianProcessKernelParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return NeuralNetworkGaussianProcessKernel.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NeuralNetworkGaussianProcessKernelParameters:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    @decorators.preprocess_inputs
    @decorators.check_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_gram(
        self,
        parameters: NeuralNetworkGaussianProcessKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix using the NNGP kernel function. If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d) if y is None, compute for x and x

        Returns: the kernel gram matrix of shape (n, m)

        """
        # if y is None, compute for x and x
        if y is None:
            y = x

        return self.ntk_kernel_function(x, y, "nngp")
