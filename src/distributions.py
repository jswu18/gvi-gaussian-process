from abc import ABC
from typing import Literal

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class Distribution(ModuleParameters, ABC):
    pass


class Gaussian(Distribution):
    mean: JaxArrayType[Literal["float64"]]
    covariance: JaxArrayType[Literal["float64"]]

    @property
    def full_covariance(self) -> bool:
        return self.covariance.ndim != self.mean.ndim


class Multinomial(Distribution):
    probabilities: JaxArrayType[Literal["float64"]]
