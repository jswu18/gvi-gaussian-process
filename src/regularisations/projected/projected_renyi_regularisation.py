import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.projected.base import ProjectedRegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class ProjectedRenyiRegularisation(ProjectedRegularisationBase):
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
    def calculate_projected_distance(
        self,
        m_p: JaxFloatType,
        c_p: JaxFloatType,
        m_q: JaxFloatType,
        c_q: JaxFloatType,
    ) -> JaxFloatType:
        return (
            jnp.log(jnp.sqrt(c_p) / jnp.sqrt(c_q))
            + (1 / (2 * (self.alpha - 1)))
            * (jnp.log(c_p / (self.alpha * c_p + (1 - self.alpha) * c_q)))
            + (1 / 2)
            * (
                (self.alpha * jnp.square(m_p - m_q))
                / (self.alpha * c_p + (1 - self.alpha) * c_q)
            )
        )
