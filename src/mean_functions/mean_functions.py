from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import flax
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.module import Module
from src.parameters.mean_functions.mean_functions import (
    ConstantFunctionParameters,
    MeanFunctionParameters,
    NeuralNetworkMeanFunctionParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class MeanFunction(Module, ABC):
    Parameters = MeanFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError

    @staticmethod
    def preprocess_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocesses the inputs of the kernel function.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
        """
        return jnp.atleast_2d(x)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self, parameters: MeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
        Calls _predict which needs to e implemented in the child class.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, 1)

        """
        x = self.preprocess_input(x)
        Module.check_parameters(parameters, self.Parameters)
        return self._predict(parameters=parameters, x=x)

    @abstractmethod
    def _predict(
        self, parameters: MeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, 1)

        """
        raise NotImplementedError


class NeuralNetworkMeanFunction(MeanFunction):
    """
    A Mean Function which is defined by a neural network.
    """

    Parameters = NeuralNetworkMeanFunctionParameters

    def __init__(
        self,
        reference_mean_function_parameters: MeanFunctionParameters,
        reference_mean_function: MeanFunction,
        neural_network: flax.linen.Module,
    ):
        """
        Using a neural network to act as the mean function of the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            reference_mean_function_parameters: the parameters of the reference mean function.
            reference_mean_function: the mean function of the reference Gaussian measure.
            neural_network: a flax linen module which takes in a design matrix of shape (n, d) and outputs a vector of shape (n, 1)
        """
        super().__init__()
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
        return self.neural_network.apply(parameters.neural_network, x)


class ConstantFunction(MeanFunction):
    Parameters = ConstantFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ConstantFunctionParameters:
        """
        Generates a Pydantic model of the parameters for Constant Functions.

        Args:
            parameters: A dictionary of the parameters for Constant Functions.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        return ConstantFunction.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ConstantFunctionParameters:
        """
        Initialise the parameters of the Constant Function using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        pass

    def _predict(
        self, parameters: ConstantFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns a constant value for all points of x.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (n,)

        """
        return parameters.constant * jnp.ones((x.shape[0],))
