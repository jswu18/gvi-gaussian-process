from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBaseParameters
from src.regularisations.base import RegularisationBase


class GeneralisedVariationalInference:
    def __init__(
        self,
        regularisation: RegularisationBase,
        empirical_risk: EmpiricalRiskBase,
    ):
        self.regularisation = regularisation
        self.empirical_risk = empirical_risk
        self._jit_compiled_calculate_loss = jax.jit(
            lambda parameters, x, y: self._calculate_loss(
                parameters=parameters, x=x, y=y
            )
        )

    def _calculate_loss(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.float64:
        return self.empirical_risk.calculate_empirical_risk(
            parameters=parameters, x=x, y=y
        ) + self.regularisation.calculate_regularisation(
            parameters=parameters,
            x=x,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_loss(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.float64:
        if not isinstance(parameters, self.regularisation.gp.Parameters):
            parameters = self.regularisation.gp.generate_parameters(parameters)
        return self._jit_compiled_calculate_loss(
            parameters.dict(),
            *(x, y),
        )
