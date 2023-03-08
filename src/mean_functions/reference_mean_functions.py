from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.mean_functions.mean_functions import MeanFunction
from src.parameters.mean_functions.mean_functions import ConstantFunctionParameters

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantFunction(MeanFunction):
    Parameters = ConstantFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(self, parameters: Union[FrozenDict, Dict]) -> Parameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return self.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self, parameters: ConstantFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
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
        return parameters.constant * jnp.ones((x.shape[0],))
