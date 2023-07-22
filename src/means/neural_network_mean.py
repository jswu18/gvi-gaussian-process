from typing import Any, Callable, Dict, Union

import flax
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.means.base import MeanBase, MeanBaseParameters

PRNGKey = Any  # pylint: disable=invalid-name


class NeuralNetworkMeanParameters(MeanBaseParameters):
    neural_network: Any  # hack fix for now


class NeuralNetworkMean(MeanBase):
    """
    A Mean Function which is defined by a neural network.
    """

    Parameters = NeuralNetworkMeanParameters

    def __init__(
        self,
        neural_network: flax.linen.Module,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Using a neural network to act as the mean function of the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            neural_network: a flax linen module which takes in a design matrix of shape (n, d) and outputs a vector of shape (n, 1)
        """
        self.neural_network = neural_network
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> NeuralNetworkMeanParameters:
        """
        Generates a Pydantic model of the parameters for Neural Network Mean Functions.

        Args:
            parameters: A dictionary of the parameters for Neural Network Mean Functions.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkMean.Parameters(neural_network=parameters["neural_network"])

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> NeuralNetworkMeanParameters:
        """
        Initialise the parameters of the ARD Kernel using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Neural Network Mean Functions.

        """
        return NeuralNetworkMeanParameters(
            neural_network=self.neural_network.init(key, jnp.zeros((1, 1)))
        )

    def _predict(
        self, parameters: NeuralNetworkMeanParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the reference mean function to the
        output of the neural network.
            - k is the number of outputs
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (n,) if k is one or (n, k)

        """
        return self.neural_network.apply(parameters.neural_network, x)
