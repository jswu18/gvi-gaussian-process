from typing import Any, Callable, Dict, Literal, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.kernels.base import KernelBase, KernelBaseParameters
from src.means.base import MeanBase, MeanBaseParameters
from src.utils.custom_types import JaxArrayType

PRNGKey = Any  # pylint: disable=invalid-name


class StochasticVariationalMeanParameters(MeanBaseParameters):
    weights: JaxArrayType[Literal["float64"]]


class StochasticVariationalMean(MeanBase):
    Parameters = StochasticVariationalMeanParameters

    def __init__(
        self,
        reference_mean_parameters: MeanBaseParameters,
        reference_mean: MeanBase,
        reference_kernel_parameters: KernelBaseParameters,
        reference_kernel: KernelBase,
        inducing_points: jnp.ndarray,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Defining the reference Gaussian measure and the reference mean function.

        Args:
            reference_mean_parameters: the parameters of the reference mean function.
            reference_mean: the mean function of the reference Gaussian measure.
        """
        self.reference_mean_parameters = reference_mean_parameters
        self.reference_mean = reference_mean
        self.reference_kernel_parameters = reference_kernel_parameters
        self.reference_kernel = reference_kernel
        self.inducing_points = inducing_points
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> StochasticVariationalMeanParameters:
        """
        Generates a Pydantic model of the parameters for SVGP Mean Functions.

        Args:
            parameters: A dictionary of the parameters for SVGP Mean Functions.

        Returns: A Pydantic model of the parameters for SVGP Mean Functions.

        """
        return StochasticVariationalMean.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> StochasticVariationalMeanParameters:
        """
        Initialise the parameters of the Constant Function using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Constant Functions.

        """
        pass

    def _predict(
        self,
        parameters: StochasticVariationalMeanParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the reference mean function to the
        product of the reference kernel gram matrix and the weights.
            - n is the number of points in x
            - d is the number of dimensions
            - m is the number of inducing points

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the provided points x of shape (n, 1).

        """
        return (
            self.reference_mean.predict(
                parameters=self.reference_mean_parameters,
                x=x,
            )
            + (
                self.reference_kernel.calculate_gram(
                    parameters=self.reference_kernel_parameters,
                    x1=x,
                    x2=x,
                    full_covariance=True,
                )
                @ parameters.weights
            ).T
        )
