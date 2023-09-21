from typing import Tuple

import jax
import jax.numpy as jnp

from src.conformal_calibration.regression.base import ConformalRegressionBase
from src.distributions import Gaussian
from src.gps.base.base import GPBaseParameters
from src.gps.base.regression_base import GPRegressionBase


class ConformalGPRegression(ConformalRegressionBase):
    def __init__(
        self,
        x_calibration: jnp.ndarray,
        y_calibration: jnp.ndarray,
        gp: GPRegressionBase,
        gp_parameters: GPBaseParameters,
    ):
        self.gp = gp
        self.gp_parameters = gp_parameters
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _predict_uncalibrated_coverage(
        self,
        x: jnp.ndarray,
        coverage: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Uses the GP to predict the lower and upper bounds of the coverage interval.
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: Tuple of lower and upper uncalibrated bounds of shape (n, 1)

        """
        gaussian = Gaussian(
            **self.gp.predict_probability(
                parameters=self.gp_parameters,
                x=x,
            ).dict()
        )
        confidence_interval_scale = jax.scipy.special.ndtri((coverage + 1) / 2)
        lower_bound = gaussian.mean - confidence_interval_scale * jnp.sqrt(
            gaussian.covariance
        )
        upper_bound = gaussian.mean + confidence_interval_scale * jnp.sqrt(
            gaussian.covariance
        )
        return jnp.atleast_2d(lower_bound), jnp.atleast_2d(upper_bound)
