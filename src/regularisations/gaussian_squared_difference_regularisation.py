import jax
import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase
from src.regularisations.schemas import RegularisationMode


class GaussianSquaredDifferenceRegularisation(RegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode = RegularisationMode.prior,
        full_covariance: bool = True,
    ):
        self.full_covariance = full_covariance
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            mode=mode,
        )

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_squared_distance(
        m_p: jnp.ndarray,
        c_p: jnp.ndarray,
        m_q: jnp.ndarray,
        c_q: jnp.ndarray,
    ) -> float:
        return jnp.float64(jnp.mean((m_p - m_q) ** 2) + jnp.mean((c_p - c_q) ** 2))

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        gaussian_p = self._calculate_regulariser_gaussian(
            x=x,
            full_covariance=self.full_covariance,
        )
        gaussian_q = self.gp.calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=self.full_covariance,
        )
        mean_p = jnp.atleast_2d(gaussian_p.mean).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        mean_q = jnp.atleast_2d(gaussian_q.mean).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        if self.full_covariance:
            covariance_p = jnp.atleast_3d(gaussian_p.covariance).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
            )
            covariance_q = jnp.atleast_3d(gaussian_q.covariance).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
            )
        else:
            covariance_p = jnp.atleast_2d(gaussian_p.covariance).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0]
            )
            covariance_q = jnp.atleast_2d(gaussian_q.covariance).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0]
            )
        return jnp.mean(
            jax.vmap(
                lambda m_p, c_p, m_q, c_q: GaussianSquaredDifferenceRegularisation.calculate_squared_distance(
                    m_p=m_p,
                    c_p=c_p,
                    m_q=m_q,
                    c_q=c_q,
                )
            )(mean_p, covariance_p, mean_q, covariance_q)
        )
