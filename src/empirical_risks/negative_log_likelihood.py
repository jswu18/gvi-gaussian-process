import jax
import jax.numpy as jnp
import jax.scipy as jsp

from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase, GPBaseParameters
from src.utils.custom_types import JaxFloatType


class NegativeLogLikelihood(EmpiricalRiskBase):
    def __init__(self, gp: GPBase):
        super().__init__(gp)

    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> JaxFloatType:
        gaussian = self.gp.calculate_prediction_gaussian(
            parameters, x=x, full_covariance=False
        )
        if self.gp.kernel.number_output_dimensions > 1:
            return jnp.float64(
                jnp.mean(
                    jax.vmap(
                        lambda y_class, mean_class, covariance_class: jnp.mean(
                            jax.vmap(
                                lambda y_, loc_, scale_: -jsp.stats.norm.logpdf(
                                    y_,
                                    loc=loc_,
                                    scale=scale_,
                                )
                            )(y_class, mean_class, jnp.sqrt(covariance_class))
                        )
                    )(y.T, gaussian.mean, gaussian.covariance)
                )
            )
        else:
            return jnp.float64(
                jnp.mean(
                    jax.vmap(
                        lambda y_, loc_, scale_: -jsp.stats.norm.logpdf(
                            y_,
                            loc=loc_,
                            scale=scale_,
                        )
                    )(
                        y.reshape(-1, 1),
                        gaussian.mean.reshape(-1, 1),
                        jnp.sqrt(gaussian.covariance.reshape(-1, 1)),
                    )
                )
            )
