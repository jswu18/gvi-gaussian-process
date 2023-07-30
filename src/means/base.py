from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.module import Module, ModuleParameters
from src.utils.custom_types import PRNGKey


class MeanBaseParameters(ModuleParameters, ABC):
    pass


class MeanBase(Module, ABC):
    Parameters = MeanBaseParameters

    def __init__(
        self,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        self.number_output_dimensions = number_output_dimensions
        super().__init__(preprocess_function=preprocess_function)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MeanBaseParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError

    @abstractmethod
    def _predict(self, parameters: MeanBaseParameters, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
            - n is the number of points in x
            - d is the number of input dimensions
            - k is the number of output dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, k)

        """
        raise NotImplementedError

    def preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocesses the inputs of the kernel function.
            - n is the number of points in x
            - d is the number of input dimensions

        Args:
            x: design matrix of shape (n, d)
        """
        return jnp.atleast_2d(self.preprocess_function(x))

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, parameters: MeanBaseParameters, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
            - n is the number of points in x
            - d is the number of input dimensions
            - k is the number of output dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (k, n)

        """
        x = self.preprocess_input(x)
        Module.check_parameters(parameters, self.Parameters)
        if self.number_output_dimensions == 1:
            return self._predict(parameters=parameters, x=x).reshape(-1)
        else:
            return self._predict(parameters=parameters, x=x).reshape(
                self.number_output_dimensions, -1
            )
