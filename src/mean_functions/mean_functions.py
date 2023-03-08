from abc import ABC, abstractmethod

import jax.numpy as jnp
import pydantic

from src.module import Module
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class MeanFunction(Module, ABC):
    Parameters = MeanFunctionParameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def predict(
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
        pass
