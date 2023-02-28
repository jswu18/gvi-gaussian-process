from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import vmap

from kernels import Kernel


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
    @staticmethod
    def kernel_func(
        parameters: FrozenDict, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        scaling = jnp.exp(parameters["log_scaling"]) ** 2
        lengthscale_matrix = jnp.diag(jnp.exp(parameters["log_lengthscales"]))
        return jnp.sum(
            scaling * jnp.exp(-0.5 * (x - y).T @ lengthscale_matrix @ (x - y))
        )


class NeuralNetworkGaussianProcessKernel(Kernel):
    def __init__(self, ntk_kernel_function: Callable):
        self.ntk_kernel_function = ntk_kernel_function

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
