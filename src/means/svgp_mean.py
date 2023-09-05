from typing import Callable, Dict, Literal, Union

import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp

from src.kernels.base import KernelBase, KernelBaseParameters
from src.means.base import MeanBase, MeanBaseParameters
from src.utils.custom_types import JaxArrayType


class SVGPMeanParameters(MeanBaseParameters):
    weights: JaxArrayType[Literal["float64"]]


class SVGPMean(MeanBase):
    Parameters = SVGPMeanParameters

    def __init__(
        self,
        regulariser_mean_parameters: MeanBaseParameters,
        regulariser_mean: MeanBase,
        regulariser_kernel_parameters: KernelBaseParameters,
        regulariser_kernel: KernelBase,
        inducing_points: jnp.ndarray,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Defining the regulariser Gaussian measure and the regulariser mean function.

        Args:
            regulariser_mean_parameters: the parameters of the regulariser mean function.
            regulariser_mean: the mean function of the regulariser Gaussian measure.
        """
        self.regulariser_mean_parameters = regulariser_mean_parameters
        self.regulariser_mean = regulariser_mean
        self.regulariser_kernel_parameters = regulariser_kernel_parameters
        self.regulariser_kernel = regulariser_kernel
        self.inducing_points = inducing_points
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> SVGPMeanParameters:
        """
        Generates a Pydantic model of the parameters for SVGP Mean Functions.

        Args:
            parameters: A dictionary of the parameters for SVGP Mean Functions.

        Returns: A Pydantic model of the parameters for SVGP Mean Functions.

        """
        return SVGPMean.Parameters(**parameters)

    def _predict(
        self,
        parameters: SVGPMeanParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict the mean function at the provided points x by adding the regulariser mean function to the
        product of the regulariser kernel gram matrix and the weights.
            - n is the number of points in x
            - d is the number of dimensions
            - m is the number of inducing points

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the provided points x of shape (n, 1).

        """
        return (
            self.regulariser_mean.predict(
                parameters=self.regulariser_mean_parameters,
                x=x,
            )
            + (
                self.regulariser_kernel.calculate_gram(
                    parameters=self.regulariser_kernel_parameters,
                    x1=x,
                    x2=x,
                    full_covariance=True,
                )
                @ parameters.weights
            ).T
        )
