import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.projected.base import ProjectedRegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class ProjectedGaussianWassersteinRegularisation(ProjectedRegularisationBase):
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

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_projected_distance(
        m_p: JaxFloatType,
        c_p: JaxFloatType,
        m_q: JaxFloatType,
        c_q: JaxFloatType,
    ) -> JaxFloatType:
        return jnp.square(m_p - m_q) + c_p + c_q - 2 * jnp.sqrt(c_p * c_q)