import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.point_wise import PointWiseKLRegularisation
from src.regularisations.point_wise.base import PointWiseRegularisationBase
from src.utils.custom_types import JaxFloatType


class PointWiseSymmetricKLRegularisation(PointWiseRegularisationBase):
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
    def calculate_point_wise_distance(
        m_p: JaxFloatType,
        c_p: JaxFloatType,
        m_q: JaxFloatType,
        c_q: JaxFloatType,
    ) -> JaxFloatType:
        return jnp.sqrt(
            PointWiseKLRegularisation.calculate_point_wise_distance(
                m_p=m_p,
                c_p=c_p,
                m_q=m_q,
                c_q=c_q,
            )
            + PointWiseKLRegularisation.calculate_point_wise_distance(
                m_p=m_q,
                c_p=c_q,
                m_q=m_p,
                c_q=c_p,
            )
        )