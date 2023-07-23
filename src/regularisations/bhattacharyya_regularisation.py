import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase
from src.utils.matrix_operations import add_diagonal_regulariser


class BhattacharyyaRegularisation(RegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        eigenvalue_regularisation: float = 1e-8,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
    ):
        self.eigenvalue_regularisation = eigenvalue_regularisation
        self.is_eigenvalue_regularisation_absolute_scale = (
            is_eigenvalue_regularisation_absolute_scale
        )
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_bhattacharyya_distance(
        m_p: jnp.ndarray,
        c_p: jnp.ndarray,
        m_q: jnp.ndarray,
        c_q: jnp.ndarray,
        eigenvalue_regularisation: float = 1e-8,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
    ) -> float:
        c_p = add_diagonal_regulariser(
            matrix=c_p,
            diagonal_regularisation=eigenvalue_regularisation,
            is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
        )
        c_q = add_diagonal_regulariser(
            matrix=c_q,
            diagonal_regularisation=eigenvalue_regularisation,
            is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
        )
        c_pq = 0.5 * (c_p + c_q)
        cholesky_decomposition_and_lower = jsp.linalg.cho_factor(c_pq)
        return 1 / 8 * (m_p - m_q).T @ jsp.linalg.cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=(m_p - m_q)
        ) + 0.5 * jnp.log(
            jnp.linalg.det(c_pq) / jnp.sqrt(jnp.linalg.det(c_p) * jnp.linalg.det(c_q))
        )

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        gaussian_p = self.regulariser.calculate_prediction_gaussian(
            parameters=self.regulariser_parameters,
            x=x,
            full_covariance=True,
        )
        mean_p, covariance_p = (
            gaussian_p.mean,
            gaussian_p.covariance,
        )
        gaussian_q = self.gp.calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=True,
        )
        mean_q, covariance_q = (
            gaussian_q.mean,
            gaussian_q.covariance,
        )
        return jnp.sum(
            jax.vmap(
                lambda m_p, c_p, m_q, c_q: BhattacharyyaRegularisation.calculate_bhattacharyya_distance(
                    m_p=m_p,
                    c_p=c_p,
                    m_q=m_q,
                    c_q=c_q,
                    eigenvalue_regularisation=self.eigenvalue_regularisation,
                    is_eigenvalue_regularisation_absolute_scale=self.is_eigenvalue_regularisation_absolute_scale,
                )
            )(
                jnp.atleast_2d(mean_p).reshape(
                    self.gp.mean.number_output_dimensions, -1
                ),
                jnp.atleast_3d(covariance_p).reshape(
                    self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
                ),
                jnp.atleast_2d(mean_q).reshape(
                    self.gp.mean.number_output_dimensions, -1
                ),
                jnp.atleast_3d(covariance_q).reshape(
                    self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
                ),
            )
        )
