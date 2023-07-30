from abc import ABC
from typing import Literal

import jax.numpy as jnp

from src.module import ModuleParameters
from src.utils.custom_types import JaxArrayType


class Distribution(ModuleParameters, ABC):
    pass


class Gaussian(Distribution):
    mean: JaxArrayType[Literal["float64"]]
    covariance: JaxArrayType[Literal["float64"]]
    full_covariance: bool = True

    @staticmethod
    def calculate_log_likelihood(
        mean: jnp.ndarray,
        covariance: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.float64:
        return


class Multinomial(Distribution):
    probabilities: JaxArrayType[Literal["float64"]]
