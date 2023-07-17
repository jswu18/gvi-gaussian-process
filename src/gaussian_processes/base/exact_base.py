from abc import ABC
from typing import Any

import jax.numpy as jnp

from src.distributions import Gaussian
from src.gaussian_processes.base.base import GaussianProcessBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.parameters.gaussian_processes.base import GaussianProcessBaseParameters

PRNGKey = Any  # pylint: disable=invalid-name


class ExactGaussianProcessBase(ABC, GaussianProcessBase):
    def __init__(
        self, mean: MeanBase, kernel: KernelBase, x: jnp.ndarray, y: jnp.ndarray
    ):
        self.x = x
        self.y = y
        super().__init__(mean=mean, kernel=kernel)

    def _calculate_prediction_distribution(
        self, parameters: GaussianProcessBaseParameters, x: jnp.ndarray
    ) -> Gaussian:
        return self.calculate_posterior_distribution(
            parameters=parameters,
            x_train=self.x,
            y_train=self.y,
            x=x,
        )
