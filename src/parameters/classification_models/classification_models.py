from abc import ABC
from typing import Literal

from src.parameters.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class ClassificationModelParameters(ModuleParameters, ABC):
    pass


class TemperedClassificationModelParameters(ModuleParameters, ABC):
    log_tempering_factors: JaxArrayType[Literal["float64"]]  # shape (k, 1)
