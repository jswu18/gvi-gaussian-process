from typing import Any, Literal

import pydantic
from jax import numpy as jnp

from src.custom_types import ArrayType
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class ApproximationMeanFunctionParameters(MeanFunctionParameters):
    pass


class StochasticVariationalGaussianProcessMeanFunctionParameters(
    ApproximationMeanFunctionParameters
):
    number_of_inducing_points: int
    weights: ArrayType[Literal["float64"]]

    @pydantic.validator("weights")
    def validate_weights_dimension(cls, weights, values):
        weights = jnp.atleast_2d(weights)
        number_of_inducing_points = values["number_of_inducing_points"]
        if weights.shape[0] != number_of_inducing_points:
            raise pydantic.ValidationError(
                f"{weights.shape[0]} must match {number_of_inducing_points=}.",
                StochasticVariationalGaussianProcessMeanFunctionParameters,
            )
        return weights


class NeuralNetworkMeanFunctionParameters(ApproximationMeanFunctionParameters):
    neural_network: Any  # hack fix for now
