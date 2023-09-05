import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.projected.base import ProjectedRegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class ProjectedHellingerRegularisation(ProjectedRegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode = RegularisationMode.prior,
    ):
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
        return 1 - jnp.sqrt(
            jnp.divide(
                2 * jnp.multiply(jnp.sqrt(c_p), jnp.sqrt(c_q)),
                c_p + c_q,
            )
        ) * jnp.exp(
            -0.25
            * jnp.divide(
                (m_p - m_q) ** 2,
                c_p + c_q,
            )
        )
