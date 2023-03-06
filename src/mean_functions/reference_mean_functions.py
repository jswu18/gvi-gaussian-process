from typing import Any

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.mean_functions.mean_functions import MeanFunction

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantFunction(MeanFunction):
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        return FrozenDict({"constant": kwargs["constant"]})

    def predict(self, parameters: FrozenDict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a constant value for all points of x.
            - n is the number of points in x
            - d is the number of dimensions

        The parameters of the mean function are:
            - constant: a constant value

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (n, 1)

        """
        return parameters["constant"] * jnp.ones((x.shape[0],))
