import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.point_wise.base import PointWiseRegularisationBase
from src.utils.custom_types import JaxFloatType


class PointWiseKLRegularisation(PointWiseRegularisationBase):
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
        return (
            jnp.log(jnp.sqrt(c_q) / jnp.sqrt(c_p))
            + (c_p + jnp.square(m_p - m_q)) / (2 * c_q)
            - 0.5
        )
