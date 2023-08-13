import jax
import jax.numpy as jnp
import jax.scipy as jsp

from src.distributions import Gaussian, Multinomial
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase, GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.gps.base.regression_base import GPRegressionBase
from src.utils.custom_types import JaxFloatType


class NegativeLogLikelihood(EmpiricalRiskBase):
    def __init__(self, gp: GPBase):
        super().__init__(gp)

    def _calculate_gaussian_log_likelihood(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> JaxFloatType:
        gaussian = Gaussian(**self.gp.predict_probability(parameters, x=x).dict())
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
                        y.reshape(-1),
                        gaussian.mean.reshape(-1),
                        jnp.sqrt(gaussian.covariance.reshape(-1)),
                    )
                )
            )

    def _calculate_multinomial_log_likelihood(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> JaxFloatType:
        multinomial = Multinomial(
            **self.gp.predict_probability(parameters=parameters, x=x).dict()
        )
        return jnp.float64(
            -jnp.sum(
                jnp.log(
                    jnp.sum(
                        jnp.multiply(
                            multinomial.probabilities,
                            y,
                        ),
                        axis=1,
                    )
                )
            )
        )

    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> JaxFloatType:
        if isinstance(self.gp, GPRegressionBase):
            return self._calculate_gaussian_log_likelihood(
                parameters=parameters,
                x=x,
                y=y,
            )
        elif isinstance(self.gp, GPClassificationBase):
            return self._calculate_multinomial_log_likelihood(
                parameters=parameters,
                x=x,
                y=y,
            )
        else:
            raise NotImplementedError(f"GP type {type(self.gp)} not implemented")
