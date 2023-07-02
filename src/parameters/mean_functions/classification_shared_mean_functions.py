from abc import ABC
from typing import Any, Literal

from src.parameters.mean_functions.mean_functions import MeanFunctionParameters
from src.utils.custom_types import JaxArrayType


class SharedMeanFunctionParameters(MeanFunctionParameters, ABC):
    pass


class NeuralNetworkSharedMeanFunctionParameters(SharedMeanFunctionParameters):
    neural_network: Any  # hack fix for now


class ConstantSharedMeanFunctionParameters(SharedMeanFunctionParameters):
    constants: JaxArrayType[Literal["float64"]]
