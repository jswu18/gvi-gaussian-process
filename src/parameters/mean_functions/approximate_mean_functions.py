from typing import Any, Literal

from src.parameters.mean_functions.mean_functions import MeanFunctionParameters
from src.utils.custom_types import JaxArrayType


class ApproximateMeanFunctionParameters(MeanFunctionParameters):
    pass


class StochasticVariationalGaussianProcessMeanFunctionParameters(
    ApproximateMeanFunctionParameters
):
    weights: JaxArrayType[Literal["float64"]]


class NeuralNetworkMeanFunctionParameters(ApproximateMeanFunctionParameters):
    neural_network: Any  # hack fix for now
