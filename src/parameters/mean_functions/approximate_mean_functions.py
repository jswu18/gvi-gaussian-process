from typing import Any, Literal

from src.parameters.custom_types import ArrayType
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class ApproximateMeanFunctionParameters(MeanFunctionParameters):
    pass


class StochasticVariationalGaussianProcessMeanFunctionParameters(
    ApproximateMeanFunctionParameters
):
    weights: ArrayType[Literal["float64"]]


class NeuralNetworkMeanFunctionParameters(ApproximateMeanFunctionParameters):
    neural_network: Any  # hack fix for now
