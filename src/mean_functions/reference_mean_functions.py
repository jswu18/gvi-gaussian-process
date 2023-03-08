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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
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
