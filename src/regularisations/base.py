from abc import ABC, abstractmethod
from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.distributions import Gaussian
from src.gps.base.base import GPBase, GPBaseParameters


class RegularisationBase(ABC):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
    ):
        self._gp = gp
        self._regulariser = regulariser
        self._regulariser_parameters = regulariser_parameters
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def gp(self) -> GPBase:
        return self._gp

    @gp.setter
    def gp(self, gp: GPBase) -> None:
        self._gp = gp
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def regulariser(self) -> GPBase:
        return self._regulariser

    @regulariser.setter
    def regulariser(self, regulariser: GPBase) -> None:
        self._regulariser = regulariser
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def regulariser_parameters(self) -> GPBaseParameters:
        return self._regulariser_parameters

    @regulariser_parameters.setter
    def regulariser_parameters(self, regulariser_parameters: GPBaseParameters) -> None:
        self._regulariser_parameters = regulariser_parameters
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @abstractmethod
    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_regularisation(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
    ) -> jnp.float64:
        if not isinstance(parameters, self.gp.Parameters):
            parameters = self.gp.generate_parameters(parameters)
        return self._jit_compiled_calculate_regularisation(
            parameters.dict(),
            x,
        )
