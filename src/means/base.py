from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.module import Module
from src.parameters.mean.base import MeanBaseParameters

PRNGKey = Any  # pylint: disable=invalid-name


class MeanBase(Module, ABC):
    Parameters = MeanBaseParameters

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

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        """
        Returns the output dimension of the mean function.
        Returns: The output dimension of the mean function.

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

    @staticmethod
    def preprocess_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocesses the inputs of the kernel function.
            - n is the number of points in x
            - d is the number of input dimensions

        Args:
            x: design matrix of shape (n, d)
        """
        return jnp.atleast_2d(x)

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

        Returns: the mean function evaluated at the points in x of shape (n, k)

        """
        x = self.preprocess_input(x)
        Module.check_parameters(parameters, self.Parameters)
        return self._predict(parameters=parameters, x=x)
