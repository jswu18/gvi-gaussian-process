from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.approximation_kernels import ApproximationKernel
from src.kernels.kernels import Kernel
from src.kernels.reference_kernels import StandardKernel
from src.mean_functions.approximation_mean_functions import \
    ApproximationMeanFunction
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
    def covariance(
        self, x: jnp.ndarray, y: jnp.ndarray, parameters: FrozenDict
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def mean(self, x: jnp.ndarray, parameters: FrozenDict) -> jnp.ndarray:
        raise NotImplementedError

    def mean_and_covariance(
        self, x: jnp.ndarray, parameters: FrozenDict
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean = self.mean(x, parameters)
        covariance = self.covariance(x, x, parameters)
        return mean, covariance

    def compute_expected_log_likelihood(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        observation_noise: float,
        parameters: FrozenDict,
    ) -> float:
        mean, covariance = self.mean_and_covariance(x, parameters)
        return (x.shape[0] / 2) * jnp.log(2 * jnp.pi * observation_noise) + (
            1 / (2 * observation_noise)
        ) * (jnp.sum((y - mean) ** 2) + jnp.trace(covariance))


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

    def mean(self, x: jnp.ndarray, parameters: FrozenDict) -> jnp.ndarray:
        gram_train = self.kernel.gram(x=self.x, parameters=parameters["kernel"])
        gram_train_test = self.kernel.gram(
            x=self.x, y=x, parameters=parameters["kernel"]
        )
        observation_noise = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters["log_observation_noise"]
        )
        cholesky_decomposition_and_lower = cho_factor(gram_train + observation_noise)
        kernel_mean = gram_train_test.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=self.y
        )
        return kernel_mean + self.mean_function.predict(
            x=x, parameters=parameters["mean_function"]
        )

    def covariance(
        self, x: jnp.ndarray, y: jnp.ndarray, parameters: FrozenDict
    ) -> jnp.ndarray:
        gram_train = self.kernel.gram(x=self.x, parameters=parameters["kernel"])
        gram_train_x = self.kernel.gram(x=self.x, y=x, parameters=parameters["kernel"])
        gram_train_y = self.kernel.gram(x=self.x, y=y, parameters=parameters["kernel"])
        gram_xy = self.kernel.gram(x=x, y=y, parameters=parameters["kernel"])
        observation_noise = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters["log_observation_noise"]
        )
        cholesky_decomposition_and_lower = cho_factor(gram_train + observation_noise)

        return gram_xy - gram_train_x.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_y
        )


class ApproximationGaussianMeasure(GaussianMeasure):
    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: ApproximationMeanFunction,
        kernel: ApproximationKernel,
    ):
        super().__init__(x, y, mean_function, kernel)
        self.kernel = kernel

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

    def mean(self, x: jnp.ndarray, parameters: FrozenDict) -> jnp.ndarray:
        return self.mean_function.predict(x=x, parameters=parameters["mean_function"])

    def covariance(
        self, x: jnp.ndarray, y: jnp.ndarray, parameters: FrozenDict
    ) -> jnp.ndarray:
        return self.kernel.gram(x=x, y=y, parameters=parameters["kernel"])
