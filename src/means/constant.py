from typing import Any, Dict, Literal, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.means.base import MeanBase, MeanBaseParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantParameters(MeanBaseParameters):
    constant: Union[JaxFloatType, JaxArrayType[Literal["float64"]]]


class Constant(MeanBase):
    Parameters = ConstantParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ConstantParameters:
        """
        Generates a Pydantic model of the parameters for Constant Functions.

        Args:
            parameters: A dictionary of the parameters for Constant Functions.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        return Constant.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ConstantParameters:
        """
        Initialise the parameters of the Constant Function using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        pass

    def _predict(self, parameters: ConstantParameters, x: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a constant value for all points of x.
            - k is the number of outputs
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (n,) if k is one or (n, k)

        """
        if isinstance(parameters.constant, jnp.ndarray):
            return jnp.tile(parameters.constant, x.shape[0]).reshape(x.shape[0], -1).T
        else:
            return parameters.constant * jnp.ones((x.shape[0],))
