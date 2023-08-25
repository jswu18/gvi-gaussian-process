import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.point_wise.base import PointWiseRegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class PointWiseRenyiRegularisation(PointWiseRegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode = RegularisationMode.prior,
        alpha: float = 0.5,
    ):
        self.alpha = alpha
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            mode=mode,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_point_wise_distance(
        self,
        m_p: JaxFloatType,
        c_p: JaxFloatType,
        m_q: JaxFloatType,
        c_q: JaxFloatType,
    ) -> JaxFloatType:
        return 0.5 * jnp.square(m_p - m_q) / (
            self.alpha * c_p + (1 - self.alpha) * c_q
        ) - (1 / (2 * self.alpha * (self.alpha - 1))) * jnp.log(
            (self.alpha * c_p + (1 - self.alpha) * c_q)
            / ((c_p**self.alpha) * (c_q ** (1 - self.alpha)))
        )
