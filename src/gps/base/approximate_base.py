from abc import ABC

import jax.numpy as jnp

from src.distributions import Gaussian
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class ApproximateGPBase(GPBase, ABC):
    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        super().__init__(mean=mean, kernel=kernel)

    def _calculate_prediction_distribution(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Gaussian:
        return self.calculate_prior_distribution(
            parameters=parameters, x=x, full_covariance=full_covariance
        )
