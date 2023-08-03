import jax
import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase


class PointWiseBhattacharyyaRegularisation(RegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
    ):
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_point_wise_bhattacharyya_distance(
        m_p: jnp.ndarray,
        c_p: jnp.ndarray,
        m_q: jnp.ndarray,
        c_q: jnp.ndarray,
    ) -> float:
        return jnp.float64(
            jnp.mean(
                jax.vmap(
                    lambda m_p_, c_p_, m_q_, c_q_: jnp.mean(
                        (
                            1
                            / 8
                            * jnp.divide(jnp.square(m_p_ - m_q_), jnp.mean(c_p_ + c_q_))
                        )
                        + (
                            1
                            / 2
                            * jnp.log(
                                jnp.divide(
                                    (c_p_ + c_q) / 2, jnp.sqrt(jnp.multiply(c_p_, c_q_))
                                )
                            )
                        )
                    )
                )(m_p, c_p, m_q, c_q)
            )
        )

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        gaussian_p = self.regulariser.calculate_prediction_gaussian(
            parameters=self.regulariser_parameters,
            x=x,
            full_covariance=False,
        )
        mean_p, covariance_p = (
            gaussian_p.mean,
            gaussian_p.covariance,
        )
        gaussian_q = self.gp.calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=False,
        )
        mean_q, covariance_q = (
            gaussian_q.mean,
            gaussian_q.covariance,
        )
        return jnp.mean(
            jax.vmap(
                lambda m_p, c_p, m_q, c_q: PointWiseBhattacharyyaRegularisation.calculate_point_wise_bhattacharyya_distance(
                    m_p=m_p,
                    c_p=c_p,
                    m_q=m_q,
                    c_q=c_q,
                )
            )(
                jnp.atleast_2d(mean_p).reshape(
                    self.gp.mean.number_output_dimensions, -1
                ),
                jnp.atleast_2d(covariance_p).reshape(
                    self.gp.mean.number_output_dimensions, x.shape[0]
                ),
                jnp.atleast_2d(mean_q).reshape(
                    self.gp.mean.number_output_dimensions, -1
                ),
                jnp.atleast_2d(covariance_q).reshape(
                    self.gp.mean.number_output_dimensions, x.shape[0]
                ),
            )
        )
