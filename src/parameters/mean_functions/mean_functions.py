from abc import ABC
from typing import Any

from src.parameters.module import ModuleParameters
from src.utils.custom_types import JaxFloatType


class MeanFunctionParameters(ModuleParameters, ABC):
    pass


class ConstantFunctionParameters(MeanFunctionParameters):
    constant: JaxFloatType


class NeuralNetworkMeanFunctionParameters(MeanFunctionParameters):
    neural_network: Any  # hack fix for now
