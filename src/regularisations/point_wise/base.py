from abc import abstractmethod

import jax
import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class PointWiseRegularisationBase(RegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode,
    ):
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            mode=mode,
        )

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def calculate_point_wise_distance(
        m_p: JaxFloatType,
        c_p: JaxFloatType,
        m_q: JaxFloatType,
        c_q: JaxFloatType,
    ) -> JaxFloatType:
        raise NotImplementedError

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        gaussian_p = self._calculate_regulariser_gaussian(
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
                lambda m_p, c_p, m_q, c_q: jnp.mean(
                    jax.vmap(
                        lambda m_p_, c_p_, m_q_, c_q_: self.calculate_point_wise_distance(
                            m_p=m_p_,
                            c_p=c_p_,
                            m_q=m_q_,
                            c_q=c_q_,
                        )
                    )(m_p, c_p, m_q, c_q)
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
