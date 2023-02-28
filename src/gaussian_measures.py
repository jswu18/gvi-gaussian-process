from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.kernels import Kernel
from src.mean_functions.mean_functions import MeanFunction


class GaussianMeasure(ABC):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: Kernel,
    ):
        """Initialising requires a kernel and data to condition the distribution."""
        self.number_of_train_points = x.shape[0]
        self.x = x
        self.y = y
        self.mean_function = mean_function
        self.kernel = kernel

    @abstractmethod
    def kernel_posterior_distribution(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        http://krasserm.github.io/2020/12/12/gaussian-processes-sparse/
        """
        raise NotImplementedError

    def mean_and_covariance(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kernel_mean, covariance = self.kernel_posterior_distribution(x, parameters)
        mean = kernel_mean + self.mean_function.predict(
            x=x, parameters=parameters["mean_function"]
        )
        return mean, covariance


class ReferenceGaussianMeasure(GaussianMeasure):
    def kernel_posterior_distribution(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass


class ApproximationGaussianMeasure(GaussianMeasure):
    def kernel_posterior_distribution(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pass

    # self.compute_kxx_shifted_cholesky_decomposition = jit(
    #     lambda kernel_parameters, log_observation_noise: self._compute_kxx_shifted_cholesky_decomposition(
    #         kernel_parameters, log_observation_noise, x, kernel
    #     )
    # )
    #
    # @staticmethod
    # def _compute_kxx_shifted_cholesky_decomposition(
    #     kernel_parameters: FrozenDict,
    #     log_observation_noise: float,
    #     x: jnp.ndarray,
    #     kernel: Kernel,
    # ) -> Tuple[jnp.ndarray, bool]:
    #     observation_variance = jnp.exp(log_observation_noise) ** 2
    #
    #     kxx = kernel.gram(kernel_parameters, x)
    #     kxx_shifted = kxx + observation_variance * jnp.eye(self.number_of_train_points)
    #     kxx_shifted_cholesky_decomposition, lower_flag = cho_factor(
    #         a=kxx_shifted, lower=True
    #     )
    #     return kxx_shifted_cholesky_decomposition, lower_flag
    #
    # def posterior_distribution(
    #     self, x: jnp.ndarray, parameters: FrozenDict
    # ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """Compute the posterior distribution for test points x.
    #     Reference: http://gaussianprocess.org/gpml/chapters/RW2.pdf
    #     """
    #     kxy = self.kernel.gram(parameters["kernel"], self.x, x)
    #     kyy = self.kernel.gram(parameters["kernel"], x)
    #     (
    #         kxx_shifted_cholesky_decomposition,
    #         lower_flag,
    #     ) = self.compute_kxx_shifted_cholesky_decomposition(
    #         kernel_parameters=parameters["kernel"],
    #         log_observation_noise=parameters["log_observation_noise"],
    #     )
    #
    #     mean = (
    #         kxy.T
    #         @ cho_solve(
    #             c_and_lower=(kxx_shifted_cholesky_decomposition, lower_flag), b=self.y
    #         )
    #     ).reshape(
    #         -1,
    #     )
    #     covariance = kyy - kxy.T @ cho_solve(
    #         (kxx_shifted_cholesky_decomposition, lower_flag), kxy
    #     )
    #     return mean, covariance
    #
    # def mean_and_covariance(
    #     self, x: jnp.ndarray, parameters: FrozenDict
    # ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     kernel_mean, covariance = self.posterior_distribution(x, parameters)
    #     mean = kernel_mean + self.mean_function.predict(
    #         x=x, parameters=parameters["mean_function"]
    #     )
    #     return mean, covariance
