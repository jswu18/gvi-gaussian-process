from abc import ABC
from typing import Any, Dict, Union

import flax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit, random

from src.kernels.reference_kernels import Kernel
from src.mean_functions.reference_mean_functions import MeanFunction
from src.parameters.gaussian_measures.reference_gaussian_measure import (
    ReferenceGaussianMeasureParameters,
)
from src.parameters.mean_functions.approximate_mean_functions import (
    ApproximateMeanFunctionParameters,
    NeuralNetworkMeanFunctionParameters,
    StochasticVariationalGaussianProcessMeanFunctionParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class ApproximateMeanFunction(MeanFunction, ABC):
    """
    Approximate mean functions which incorporate the mean function of the reference mean function into its prediction.
    The approximate mean function itself may or may not be defined with respect to the reference Gaussian measure.
    """

    Parameters = ApproximateMeanFunctionParameters

    def __init__(
        self,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        reference_mean_function: MeanFunction,
    ):
        """
        Defining the reference Gaussian measure and the reference mean function.

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_mean_function: the mean function of the reference Gaussian measure.
        """
        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )

        # define a jit-compiled version of the reference mean function using the reference mean function parameters
        self.reference_mean_func = jit(
            lambda x: reference_mean_function.predict(
                parameters=reference_gaussian_measure_parameters.mean_function, x=x
            )
        )


class StochasticVariationalGaussianProcessMeanFunction(ApproximateMeanFunction):
    """
    The mean function of a stochastic variational Gaussian process, defined with respect to the reference kernel.
    """

    Parameters = StochasticVariationalGaussianProcessMeanFunctionParameters

    def __init__(
        self,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        reference_mean_function: MeanFunction,
        reference_kernel: Kernel,
        inducing_points: jnp.ndarray,
    ):
        """
        Defining the reference Gaussian measure, the reference mean function, the reference kernel and the inducing points.
        - m is the number of inducing points
        - d is the number of dimensions

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_mean_function: the mean function of the reference Gaussian measure.
            reference_kernel: the reference kernel of the reference Gaussian measure.
            inducing_points: the inducing points of the stochastic variational Gaussian process of shape (m, d).
        """
        super().__init__(reference_gaussian_measure_parameters, reference_mean_function)
        self.inducing_points = inducing_points
        self.number_of_inducing_points = inducing_points.shape[0]

        # define a jit-compiled version of the reference kernel gram matrix using the reference kernel parameters
        self.calculate_reference_gram = jit(
            lambda x: reference_kernel.calculate_gram(
                parameters=reference_gaussian_measure_parameters.kernel,
                x=x,
                y=inducing_points,
            )
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> StochasticVariationalGaussianProcessMeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for Stochastic Variational Gaussian Process Mean Functions.

        Args:
            parameters: A dictionary of the parameters for Stochastic Variational Gaussian Process Mean Functions.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Mean Functions.

        """
        return StochasticVariationalGaussianProcessMeanFunction.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> StochasticVariationalGaussianProcessMeanFunctionParameters:
        """
        Initialise the weight parameters from a normal distribution using a random key and
        scaling by the number of inducing points

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Stochastic Variational Gaussian Process Mean Functions.

        """
        weights = random.normal(key, (self.number_of_inducing_points, 1)) / (
            self.number_of_inducing_points
        )
        return StochasticVariationalGaussianProcessMeanFunction.Parameters(
            weights=weights
        )

    def _predict(
        self,
        parameters: StochasticVariationalGaussianProcessMeanFunctionParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the reference mean function to the
        product of the reference kernel gram matrix and the weights.
            - n is the number of points in x
            - d is the number of dimensions
            - m is the number of inducing points

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the provided points x of shape (n, 1).

        """
        return (
            self.reference_mean_func(x)
            + (self.calculate_reference_gram(x) @ parameters.weights).T
        ).reshape(-1)


class NeuralNetworkMeanFunction(ApproximateMeanFunction):
    """
    A Mean Function which is defined by a neural network.
    """

    Parameters = NeuralNetworkMeanFunctionParameters

    def __init__(
        self,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        reference_mean_function: MeanFunction,
        neural_network: flax.linen.Module,
    ):
        """
        Using a neural network to act as the mean function of the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_mean_function: the mean function of the reference Gaussian measure.
            neural_network: a flax linen module which takes in a design matrix of shape (n, d) and outputs a vector of shape (n, 1)
        """
        super().__init__(reference_gaussian_measure_parameters, reference_mean_function)
        self.neural_network = neural_network

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NeuralNetworkMeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Mean Functions.

        Args:
            parameters: A dictionary of the parameters for Neural Network Mean Functions.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkMeanFunction.Parameters(
            neural_network=parameters["neural_network"]
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NeuralNetworkMeanFunctionParameters:
        """
        Initialise the parameters of the ARD Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkMeanFunctionParameters(
            neural_network=self.neural_network.init(key, jnp.zeros((1, 1)))
        )

    def _predict(
        self, parameters: NeuralNetworkMeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the reference mean function to the
        output of the neural network.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the provided points x of shape (n, 1).

        """
        return self.reference_mean_func(x) + self.neural_network.apply(
            parameters.neural_network, x
        )
