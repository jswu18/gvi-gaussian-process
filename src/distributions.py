from abc import ABC
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Distribution(ABC):
    pass


@dataclass
class Gaussian(Distribution):
    mean: jnp.ndarray
    covariance: jnp.ndarray
    full_covariance: bool = True


@dataclass
class Multinomial(Distribution):
    probabilities: jnp.ndarray
