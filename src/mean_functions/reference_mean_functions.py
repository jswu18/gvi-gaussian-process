from typing import Any, Dict

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src import decorators
from src.mean_functions.mean_functions import MeanFunction

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantFunction(MeanFunction):
    parameter_keys = {
        "constant": float,
    }

    @decorators.common.check_parameters(parameter_keys)
    def initialise_parameters(self, parameters: Dict[str, type]) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            parameters: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        return self._initialise_parameters(parameters=parameters)

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

    @decorators.common.check_parameters(parameter_keys)
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
