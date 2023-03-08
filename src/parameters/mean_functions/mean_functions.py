from abc import ABC

from src.parameters.module import ModuleParameters


class MeanFunctionParameters(ModuleParameters, ABC):
    pass


class ConstantFunctionParameters(MeanFunctionParameters):
    constant: float
