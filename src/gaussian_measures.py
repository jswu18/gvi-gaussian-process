from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximation_kernels import ApproximationKernel
from src.kernels.kernels import Kernel
from src.kernels.reference_kernels import StandardKernel
from src.mean_functions.approximation_mean_functions import ApproximationMeanFunction
from src.mean_functions.mean_functions import MeanFunction
from src.module import Module

PRNGKey = Any  # pylint: disable=invalid-name


class GaussianMeasure(Module, ABC):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: Kernel,
    ):
        self.number_of_train_points = x.shape[0]
        self.x = x
        self.y = y
        self.mean_function = mean_function
        self.kernel = kernel

    @abstractmethod
    def mean_and_covariance(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


class ReferenceGaussianMeasure(GaussianMeasure):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: StandardKernel,
    ):
        super().__init__(x, y, mean_function, kernel)

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        return FrozenDict(
            {
                "log_observation_noise": random.normal(key),
                "mean_function": self.mean_function.initialise_random_parameters(key),
                "kernel": self.kernel.initialise_random_parameters(key),
            }
        )

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return FrozenDict(
            {
                "log_observation_noise": kwargs["log_observation_noise"],
                "mean_function": self.mean_function.initialise_parameters(
                    **kwargs["mean_function"]
                ),
                "kernel": self.kernel.initialise_parameters(**kwargs["kernel"]),
            }
        )

    def mean_and_covariance(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        kernel_mean, covariance = self.kernel_posterior_distribution(x, parameters)
        mean = kernel_mean + self.mean_function.predict(
            x=x, parameters=parameters["mean_function"]
        )
        return mean, covariance

    def kernel_posterior_distribution(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gram_train = self.kernel.gram(x=self.x, parameters=parameters["kernel"])
        gram_train_test = self.kernel.gram(
            x=self.x, y=x, parameters=parameters["kernel"]
        )
        gram_test = self.kernel.gram(x=x, parameters=parameters["kernel"])
        observation_noise = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters["log_observation_noise"]
        )
        cholesky_decomposition_and_lower = cho_factor(gram_train + observation_noise)

        mean = gram_train_test.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=self.y
        )
        covariance = gram_test - gram_train_test.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_test
        )
        return mean, covariance


class ApproximationGaussianMeasure(GaussianMeasure):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: ApproximationMeanFunction,
        kernel: ApproximationKernel,
    ):
        super().__init__(x, y, mean_function, kernel)

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        return FrozenDict(
            {
                "mean_function": self.mean_function.initialise_random_parameters(key),
                "kernel": self.kernel.initialise_random_parameters(key),
            }
        )

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return FrozenDict(
            {
                "mean_function": self.mean_function.initialise_parameters(
                    **kwargs["mean_function"]
                ),
                "kernel": self.kernel.initialise_parameters(**kwargs["kernel"]),
            }
        )

    def mean_and_covariance(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean = self.mean_function.predict(x=x, parameters=parameters["mean_function"])
        covariance = self.kernel.gram(x=x, parameters=parameters["kernel"])
        return mean, covariance
