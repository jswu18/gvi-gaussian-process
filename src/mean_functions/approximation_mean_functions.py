from abc import ABC
from typing import Any

import flax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit, random

from src.kernels.reference_kernels import Kernel
from src.mean_functions.reference_mean_functions import MeanFunction

PRNGKey = Any  # pylint: disable=invalid-name


class ApproximationMeanFunction(MeanFunction, ABC):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_mean_function: MeanFunction,
    ):
        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )
        self.reference_mean_func = jit(
            lambda x: reference_mean_function.predict(
                parameters=reference_gaussian_measure_parameters["mean_function"], x=x
            )
        )


class StochasticVariationalGaussianProcessMeanFunction(ApproximationMeanFunction):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_mean_function: MeanFunction,
        reference_kernel: Kernel,
        inducing_points: jnp.ndarray,
    ):
        super().__init__(reference_gaussian_measure_parameters, reference_mean_function)
        # self.inducing_points = inducing_points
        self.number_of_inducing_points = inducing_points.shape[0]
        self.reference_kernel_gram = jit(
            lambda x: reference_kernel.gram(
                parameters=reference_gaussian_measure_parameters["kernel"],
                x=x,
                y=inducing_points,
            )
        )

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        weights = random.normal(key, (self.number_of_inducing_points, 1)) / (
            self.number_of_inducing_points**2
        )
        return FrozenDict({"weights": weights})

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        assert "weights" in kwargs, "weights must be provided"
        weights = jnp.atleast_2d(kwargs["weights"])
        assert weights.shape == (
            self.number_of_inducing_points,
            1,
        ), f"weights must have shape ({self.number_of_inducing_points}, 1), the shape provided was {weights.shape}"
        return FrozenDict({"weights": weights})

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return (
            self.reference_mean_func(x)
            + (self.reference_kernel_gram(x) @ parameters["weights"]).T
        ).reshape(-1)


class NeuralNetworkMeanFunction(ApproximationMeanFunction):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_mean_function: MeanFunction,
        neural_network: flax.linen.Module,
    ):
        super().__init__(reference_gaussian_measure_parameters, reference_mean_function)
        self.neural_network = neural_network

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        return self.neural_network.init(key, jnp.zeros((1, 1)))

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return FrozenDict(kwargs)

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return self.reference_mean_func(x) + self.neural_network.apply(parameters, x)
