from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.means.base import MeanBase, MeanBaseParameters
from src.utils.custom_types import PRNGKey


class MultiOutputMeanParameters(MeanBaseParameters):
    means: List[MeanBaseParameters]


class MultiOutputMean(MeanBase):
    Parameters = MultiOutputMeanParameters

    def __init__(self, means: List[MeanBase]):
        assert all(mean.number_output_dimensions == 1 for mean in means)
        self.means = means
        super().__init__(
            number_output_dimensions=len(means),
            preprocess_function=None,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> MultiOutputMeanParameters:
        """
        Generates a Pydantic model of the parameters for a multi-output mean function.

        Args:
            parameters: A dictionary of the parameters for a multi-output mean function.

        Returns: A Pydantic model of the parameters fo a multi-output mean function.

        """
        assert len(parameters["means"]) == self.number_output_dimensions
        return MultiOutputMean.Parameters(**parameters)

    def _predict(
        self,
        parameters: MultiOutputMeanParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the prior gram matrix of multiple kernels.
            - k is the number of kernels
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)

        Returns: the kernel a stacked gram matrix of shape (k, n)
        """
        return jnp.array(
            [
                kernel_.predict(
                    parameters=parameters_,
                    x=x,
                )
                for kernel_, parameters_ in zip(self.means, parameters.means)
            ]
        )
