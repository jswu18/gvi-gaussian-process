from abc import ABC, abstractmethod
from typing import Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.module import Module
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class MeanFunction(Module, ABC):
    Parameters = MeanFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MeanFunctionParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        raise NotImplementedError

    @staticmethod
    def preprocess_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocesses the inputs of the kernel function.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
        """
        return jnp.atleast_2d(x)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self, parameters: MeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
        Calls _predict which needs to e implemented in the child class.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, 1)

        """
        x = self.preprocess_input(x)
        Module.check_parameters(parameters, self.Parameters)
        return self._predict(parameters, x)

    @abstractmethod
    def _predict(
        self, parameters: MeanFunctionParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the mean function at the given points.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, 1)

        """
        raise NotImplementedError
