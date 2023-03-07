from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import vmap

from src import decorators
from src.common import compose_decorators
from src.kernels.kernels import Kernel

PRNGKey = Any  # pylint: disable=invalid-name


class StandardKernel(Kernel, ABC):
    parameter_keys = NotImplemented
    """
    Kernel class for kernels that can be easily defined with a kernel function evaluated at a single pair of points.
    """

    @staticmethod
    @decorators.common.check_parameters(parameter_keys)
    @abstractmethod
    def calculate_kernel(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
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
    parameter_keys = {"log_scaling": jnp.float64, "log_lengthscales": jnp.ndarray}

    @decorators.common.check_parameters(parameter_keys)
    def initialise_parameters(self, parameters: Dict[str, type]) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            parameters: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        return self._initialise_parameters(parameters=parameters)

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
    @decorators.common.check_parameters(parameter_keys)
    def calculate_kernel(
        self, parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
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
        scaling = jnp.exp(jnp.atleast_1d(parameters["log_scaling"])) ** 2
        lengthscale_matrix = jnp.diag(
            jnp.exp(jnp.atleast_1d(parameters["log_lengthscales"]))
        )
        return jnp.sum(
            scaling * jnp.exp(-0.5 * (x - y) @ lengthscale_matrix @ (x - y).T)
        ).astype(jnp.float64)

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
    @decorators.common.check_parameters(parameter_keys)
    def calculate_gram(
        self,
        parameters: FrozenDict,
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
    parameter_keys = {}

    def __init__(self, ntk_kernel_function: Callable):
        """
        A wrapper class for the kernel function provided by the NTK package.
        Args:
            ntk_kernel_function: The kernel function provided by the NTK package.
        """
        self.ntk_kernel_function = ntk_kernel_function

    @decorators.common.check_parameters(parameter_keys)
    def initialise_parameters(self, parameters: Dict[str, type]) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            parameters: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        return self._initialise_parameters(parameters=parameters)

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
    @decorators.common.check_parameters(parameter_keys)
    def calculate_gram(
        self,
        parameters: FrozenDict,
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
        return self.ntk_kernel_function(x, y, "nngp")
