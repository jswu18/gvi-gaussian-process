from abc import ABC

import jax.numpy as jnp

from src.distributions import Gaussian
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class GPRegressionBase(ABC, GPBase):
    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        super().__init__(mean=mean, kernel=kernel)

    def _predict_probability(
        self, parameters: GPBaseParameters, x: jnp.ndarray
    ) -> Gaussian:
        return self._calculate_prediction_distribution(parameters=parameters, x=x)
