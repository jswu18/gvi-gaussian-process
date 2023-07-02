from abc import ABC
from typing import Any, Dict, Union

import flax
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.mean_functions.mean_functions import MeanFunction, NeuralNetworkMeanFunction
from src.parameters.mean_functions.classification_shared_mean_functions import (
    ConstantSharedMeanFunctionParameters,
    NeuralNetworkSharedMeanFunctionParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class SharedMeanFunction(MeanFunction, ABC):
    pass


class NeuralNetworkSharedMeanFunction(SharedMeanFunction):
    """
    A Mean Function which is defined by a neural network.
    """

    Parameters = NeuralNetworkSharedMeanFunctionParameters

    def __init__(
        self,
        neural_network: flax.linen.Module,
    ):
        """
        Using a neural network to act as the mean function of the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            neural_network: a flax linen module which takes in a design matrix of shape (n, d) and outputs a vector of shape (n, 1)
        """
        super().__init__()
        self.neural_network = neural_network

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NeuralNetworkSharedMeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Mean Functions.

        Args:
            parameters: A dictionary of the parameters for Neural Network Mean Functions.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkSharedMeanFunction.Parameters(
            neural_network=parameters["neural_network"]
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NeuralNetworkSharedMeanFunctionParameters:
        """
        Initialise the parameters of the ARD Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkSharedMeanFunctionParameters(
            neural_network=self.neural_network.init(key, jnp.zeros((1, 1)))
        )

    def _predict(
        self, parameters: NeuralNetworkSharedMeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the reference mean function to the
        output of the neural network.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the provided points x of shape (k, n).

        """
        return (
            self.neural_network.apply(parameters.neural_network, x)
            .reshape(x.shape[0], -1)
            .T
        )


class ConstantSharedMeanFunction(MeanFunction):
    Parameters = ConstantSharedMeanFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ConstantSharedMeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for Constant Functions.

        Args:
            parameters: A dictionary of the parameters for Constant Functions.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        return ConstantSharedMeanFunction.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ConstantSharedMeanFunctionParameters:
        """
        Initialise the parameters of the Constant Function using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        pass

    def _predict(
        self, parameters: ConstantSharedMeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns a constant output vector for all points of x.
            - k is the number of classes
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (k, n)

        """
        return jnp.tile(parameters.constants, x.shape[0]).reshape(x.shape[0], -1).T
