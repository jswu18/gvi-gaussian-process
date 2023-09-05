from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.regularisations.base import RegularisationBase


class GeneralisedVariationalInference:
    """
    Generalised variational inference following the framework from:
        Jeremias Knoblauch, Jack Jewson, and Theodoros Damoulas. An optimization-centric view on
            bayes’ rule: Reviewing and generalizing variational inference. Journal of Machine Learning
            Research, 23(132):1–109, 2022.
    """

    def __init__(
        self,
        regularisation: RegularisationBase,
        empirical_risk: EmpiricalRiskBase,
    ):
        self._regularisation = regularisation
        self._empirical_risk = empirical_risk
        self._jit_compiled_calculate_loss = jax.jit(
            lambda parameters, x, y: self._calculate_loss(
                parameters=parameters, x=x, y=y
            )
        )

    @property
    def regularisation(self) -> RegularisationBase:
        return self._regularisation

    @regularisation.setter
    def regularisation(self, regularisation: RegularisationBase) -> None:
        """
        Set the regularisation. This will recompile the jitted function.
        Args:
            regularisation: The new regularisation.

        """
        self._regularisation = regularisation
        self._jit_compiled_calculate_loss = jax.jit(
            lambda parameters, x, y: self._calculate_loss(
                parameters=parameters, x=x, y=y
            )
        )

    @property
    def empirical_risk(self) -> EmpiricalRiskBase:
        return self._empirical_risk

    @empirical_risk.setter
    def empirical_risk(self, empirical_risk: EmpiricalRiskBase) -> None:
        """
        Set the empirical risk. This will recompile the jitted function.
        Args:
            empirical_risk: The new empirical risk.

        """
        self._empirical_risk = empirical_risk
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
        """
        Calculate the GVI objective. This is the empirical risk and the regularisation.
        Args:
            parameters: The parameters of the GP.
            x: The input data.
            y: The response data.

        Returns: The GVI objective.

        """
        return self.empirical_risk.calculate_empirical_risk(
            parameters=parameters, x=x, y=y
        ) + self.regularisation.calculate_regularisation(
            parameters=parameters,
            x=x,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_loss(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.float64:
        """
        Calculate the GVI objective. This is the empirical risk and the regularisation.
        Calls the jitted function.
        Args:
            parameters: The parameters of the GP.
            x: The input data.
            y: The response data.

        Returns: The GVI objective.

        """
        if not isinstance(parameters, self.regularisation.gp.Parameters):
            parameters = self.regularisation.gp.generate_parameters(parameters)
        return self._jit_compiled_calculate_loss(
            parameters.dict(),
            *(x, y),
        )
