import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.regularisations.projected.base import ProjectedRegularisationBase
from src.regularisations.schemas import RegularisationMode
from src.utils.custom_types import JaxFloatType


class ProjectedKLRegularisation(ProjectedRegularisationBase):
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
    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_projected_distance(
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
