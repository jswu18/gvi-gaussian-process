from typing import Any, Callable, Dict, Literal, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.means.base import MeanBase, MeanBaseParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType

PRNGKey = Any  # pylint: disable=invalid-name


class ConstantMeanParameters(MeanBaseParameters):
    constant: Union[JaxFloatType, JaxArrayType[Literal["float64"]]]


class ConstantMean(MeanBase):
    Parameters = ConstantMeanParameters

    def __init__(
        self,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ConstantMeanParameters:
        """
        Generates a Pydantic model of the parameters for ConstantMean Functions.

        Args:
            parameters: A dictionary of the parameters for ConstantMean Functions.

        Returns: A Pydantic model of the parameters for ConstantMean Functions.

        """
        if self.number_output_dimensions > 1:
            assert parameters["constant"].shape[0] == self.number_output_dimensions
        return ConstantMean.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ConstantMeanParameters:
        """
        Initialise the parameters of the ConstantMean Function using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for ConstantMean Functions.

        """
        pass

    def _predict(
        self, parameters: ConstantMeanParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns a constant value for all points of x.
            - k is the number of outputs
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: a constant vector of shape (n, k)

        """
        return jnp.tile(parameters.constant, x.shape[0])
