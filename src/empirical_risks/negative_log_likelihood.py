import jax.numpy as jnp

from src.distributions import Gaussian
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase, GPBaseParameters


class NegativeLogLikelihood(EmpiricalRiskBase):
    def __init__(self, gp: GPBase):
        super().__init__(gp)

    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        gaussian = self.gp.calculate_prediction_gaussian(
            parameters, x=x, full_covariance=False
        )
        return -Gaussian.calculate_log_likelihood(
            mean=gaussian.mean,
            covariance_diagonal=gaussian.covariance.reshape(gaussian.mean.shape),
            y=y.reshape(gaussian.mean.shape),
        )
