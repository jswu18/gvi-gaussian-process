from abc import ABC

from src.utils.custom_types import JaxFloatType
from src.parameters.module import ModuleParameters


class MeanFunctionParameters(ModuleParameters, ABC):
    pass


class ConstantFunctionParameters(MeanFunctionParameters):
    constant: JaxFloatType
