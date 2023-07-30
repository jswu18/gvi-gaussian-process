from abc import ABC, abstractmethod
from typing import Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.gps.base.base import GPBase, GPBaseParameters
from src.utils.jit_compiler import JitCompiler


class EmpiricalRiskBase(ABC):
    def __init__(self, gp: GPBase):
        self.gp = gp
        self._jit_compiled_calculate_empirical_risk = JitCompiler(
            self._calculate_empirical_risk
        )

    @abstractmethod
    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
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
