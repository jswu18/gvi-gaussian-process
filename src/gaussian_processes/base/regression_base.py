from abc import ABC

import jax.numpy as jnp

from src.distributions import Gaussian
from src.gaussian_processes.base.base import GaussianProcessBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.parameters.gaussian_processes.base import GaussianProcessBaseParameters


class GaussianProcessRegressionBase(ABC, GaussianProcessBase):
    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        super().__init__(mean=mean, kernel=kernel)

    def _predict_probability(
        self, parameters: GaussianProcessBaseParameters, x: jnp.ndarray
    ) -> Gaussian:
        return self._calculate_prediction_distribution(parameters=parameters, x=x)
