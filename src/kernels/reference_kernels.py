from abc import ABC, abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import vmap

from src.kernels.kernels import Kernel

PRNGKey = Any  # pylint: disable=invalid-name


class StandardKernel(Kernel, ABC):
    """
    Kernel class for kernels that can be easily defined with a kernel function evaluated at a single pair of points.
    """

    @staticmethod
    @abstractmethod
    def calculate_kernel(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
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

    def calculate_gram(
        self, parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
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
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        return vmap(
            lambda x_: vmap(lambda y_: self.calculate_kernel(parameters, x_, y_))(y)
        )(x)


class ARDKernel(StandardKernel):
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

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        return FrozenDict(
            {
                "log_scaling": kwargs["log_scaling"],
                "log_lengthscales": kwargs["log_lengthscales"],
            }
        )

    @staticmethod
    def calculate_kernel(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """
        The ARD kernel function defined as:
        k(x, y) = scaling * exp(-0.5 * (x - y)^T @ diag(1 / lengthscales) @ (x - y))
        where:
        - scaling is a scalar (log_scaling in the parameters)
        - lengthscales is a vector of length d (log_lengthscales in the parameters
        - x and y are vectors of length d, the number of dimensions
        - n is the number of points in x
        - m is the number of points in y
        - d is the number of dimensions

        The parameters of the kernel are:
        - log_scaling: the log of the scaling parameter, a float
        - log_lengthscales: the log of the lengthscales parameter of shape (d, 1)

        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (1, d)
            y: design matrix of shape (1, d)

        Returns: the ARD kernel function evaluated at x and y

        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        scaling = jnp.exp(jnp.atleast_1d(parameters["log_scaling"])) ** 2
        lengthscale_matrix = jnp.diag(
            jnp.exp(jnp.atleast_1d(parameters["log_lengthscales"]))
        )
        return jnp.sum(
            scaling * jnp.exp(-0.5 * (x - y).T @ lengthscale_matrix @ (x - y))
        )


class NeuralNetworkGaussianProcessKernel(Kernel):
    def __init__(self, ntk_kernel_function: Callable):
        """
        A wrapper class for the kernel function provided by the NTK package.
        Args:
            ntk_kernel_function: The kernel function provided by the NTK package.
        """
        self.ntk_kernel_function = ntk_kernel_function

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

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    def calculate_gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix using the NNGP kernel function. If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)

        """
        # compute k(x, x) if y is None
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        assert (
            x.shape[1] == y.shape[1]
        ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"

        return self.ntk_kernel_function(x, y, "nngp")
