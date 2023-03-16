from abc import ABC

from src.parameters.module import ModuleParameters
from src.utils.custom_types import JaxFloatType


class GaussianMeasureParameters(ModuleParameters, ABC):
    pass


class TemperedGaussianMeasureParameters(GaussianMeasureParameters):
    log_tempering_factor: JaxFloatType
