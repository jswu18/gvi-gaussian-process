from abc import ABC, abstractmethod
from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.gps.base.base import GPBase, GPBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG


class EmpiricalRiskBase(ABC):
    """
    A base class for all empirical risks.
    """

    def __init__(self, gp: GPBase):
        self._gp = gp
        self._jit_compiled_calculate_empirical_risk = jax.jit(
            lambda parameters, x, y: self._calculate_empirical_risk(
                parameters=parameters,
                x=x,
                y=y,
            )
        )

    @property
    def gp(self) -> GPBase:
        return self._gp

    @gp.setter
    def gp(self, gp: GPBase) -> None:
        """
        Set the GP.
        This will recompile the JIT compiled function.
        Args:
            gp:

        Returns:

        """
        self._gp = gp
        self._jit_compiled_calculate_empirical_risk = jax.jit(
            lambda parameters, x, y: self._calculate_empirical_risk(
                parameters=parameters,
                x=x,
                y=y,
            )
        )

    @abstractmethod
    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        raise NotImplementedError

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_empirical_risk(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        if not isinstance(parameters, self.gp.Parameters):
            parameters = self.gp.generate_parameters(parameters)
        return self._jit_compiled_calculate_empirical_risk(
            parameters.dict(),
            *(x, y),
        )
