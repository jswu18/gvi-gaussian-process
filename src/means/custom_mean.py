from typing import Any, Callable, Dict, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.means.base import MeanBase, MeanBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG


class CustomMeanParameters(MeanBaseParameters):
    custom: Any


class CustomMean(MeanBase):
    """
    A Mean Function which is defined by a custom function.
    """

    Parameters = CustomMeanParameters

    def __init__(
        self,
        mean_function: Callable[[Any, jnp.ndarray], jnp.ndarray],
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.mean_function = mean_function
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> CustomMeanParameters:
        return CustomMean.Parameters(custom=parameters["custom"])

    def _predict(self, parameters: CustomMeanParameters, x: jnp.ndarray) -> jnp.ndarray:
        return self.mean_function(parameters.custom, x)
