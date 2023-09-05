from abc import ABC
from typing import Tuple

import jax.numpy as jnp

from src.distributions import Gaussian
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class GPRegressionBase(GPBase, ABC):
    """
    A base class for all GP regression models. All GP regression model classes will inheret this ABC.
    """

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        super().__init__(mean=mean, kernel=kernel)

    def _construct_distribution(
        self,
        probabilities: Tuple[jnp.ndarray, jnp.ndarray],
        full_covariance: bool = False,
    ) -> Gaussian:
        mean, covariance = probabilities
        return Gaussian(
            mean=mean, covariance=covariance, full_covariance=full_covariance
        )

    def _predict_probability(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=False,
        )
