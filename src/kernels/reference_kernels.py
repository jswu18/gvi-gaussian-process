from abc import ABC, abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import vmap

from src.kernels.kernels import Kernel

PRNGKey = Any  # pylint: disable=invalid-name


class StandardKernel(Kernel, ABC):
    @staticmethod
    @abstractmethod
    def kernel_func(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    def gram(
        self, parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        return vmap(
            lambda x_: vmap(lambda y_: self.kernel_func(parameters, x_, y_))(y)
        )(x)


class ARDKernel(StandardKernel):
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return FrozenDict(
            {
                "log_scaling": kwargs["log_scaling"],
                "log_lengthscales": kwargs["log_lengthscales"],
            }
        )

    @staticmethod
    def kernel_func(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
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
        self.ntk_kernel_function = ntk_kernel_function

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        pass

    def gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
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
