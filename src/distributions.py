from abc import ABC
from typing import Literal

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class Distribution(ModuleParameters, ABC):
    """
    A base class for all distribution parameters. All distribution classes will inheret this ABC.
    """

    pass


class Gaussian(Distribution):
    """
    A Gaussian distribution with mean and covariance.
    """

    mean: JaxArrayType[Literal["float64"]]
    covariance: JaxArrayType[Literal["float64"]]

    @property
    def full_covariance(self) -> bool:
        """
        Indicator for whether the full covariance is stored. Otherwise, just the diagonal is stored.
        Returns: True if the full covariance is stored, False otherwise.

        """
        return self.covariance.ndim != self.mean.ndim


class Multinomial(Distribution):
    """
    A Multinomial distribution with probabilities.
    """

    probabilities: JaxArrayType[Literal["float64"]]
