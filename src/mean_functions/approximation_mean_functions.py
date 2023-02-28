from abc import ABC

import flax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit

from src.kernels.reference_kernels import Kernel
from src.mean_functions.reference_mean_functions import MeanFunction


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
                reference_gaussian_measure_parameters["mean"], x
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
        self.inducing_points = inducing_points
        self.reference_kernel_gram = jit(
            lambda x: reference_kernel.gram(
                reference_gaussian_measure_parameters["kernel"], inducing_points, x
            )
        )

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return (
            self.reference_mean_func(x)
            + self.reference_kernel_gram(x) @ parameters["weights"]
        )


class NeuralNetworkMeanFunction(ApproximationMeanFunction):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_mean_function: MeanFunction,
        neural_network: flax.linen.Module,
    ):
        super().__init__(reference_gaussian_measure_parameters, reference_mean_function)
        self.neural_network = neural_network

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        return self.reference_mean_func(x) + self.neural_network.apply(parameters, x)
