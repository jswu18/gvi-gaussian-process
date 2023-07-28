from abc import ABC
from typing import Any, Tuple

import jax.numpy as jnp

from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.base import KernelBase
from src.means.base import MeanBase

PRNGKey = Any  # pylint: disable=invalid-name


class ExactGPBase(GPBase, ABC):
    def __init__(
        self, mean: MeanBase, kernel: KernelBase, x: jnp.ndarray, y: jnp.ndarray
    ):
        self.x = x
        self.y = y
        GPBase.__init__(self, mean=mean, kernel=kernel)

    def _calculate_prediction_gaussian(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.calculate_posterior(
            parameters=parameters,
            x_train=self.x,
            y_train=self.y,
            x=x,
            full_covariance=full_covariance,
        )

    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        return self.calculate_posterior_covariance(
            parameters=parameters,
            x_train=self.x,
            y_train=self.y,
            x=x,
            full_covariance=full_covariance,
        )
