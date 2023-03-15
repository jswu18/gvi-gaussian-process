from abc import ABC

from src.parameters.module import ModuleParameters
from src.utils.custom_types import JaxFloatType


class MeanFunctionParameters(ModuleParameters, ABC):
    pass


class ConstantFunctionParameters(MeanFunctionParameters):
    constant: JaxFloatType
