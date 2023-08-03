from abc import ABC
from typing import Literal

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class Distribution(ModuleParameters, ABC):
    pass


class Gaussian(Distribution):
    mean: JaxArrayType[Literal["float64"]]
    covariance: JaxArrayType[Literal["float64"]]
    full_covariance: bool = True


class Multinomial(Distribution):
    probabilities: JaxArrayType[Literal["float64"]]
