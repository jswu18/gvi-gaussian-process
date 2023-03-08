from typing import Dict, Union

import pydantic
from flax.core import FrozenDict
from jax import numpy as jnp

from src.gaussian_measures.gaussian_measure import GaussianMeasure, PRNGKey
from src.kernels.approximate_kernels import ApproximateKernel
from src.mean_functions.approximate_mean_functions import ApproximateMeanFunction
from src.parameters.gaussian_measures.approximate_gaussian_measure import (
    ApproximateGaussianMeasureParameters,
)
from src.utils import decorators


class ApproximateGaussianMeasure(GaussianMeasure):
    """
    A Gaussian measure which uses an approximate to the kernel and mean function. These are defined with respect
    to reference Gaussian measures.
    """

    Parameters = ApproximateGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: ApproximateMeanFunction,
        kernel: ApproximateKernel,
    ):
        """
        Defining the training data, the mean function and the kernel for the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: the training inputs of shape (n, d)
            y: the training outputs of shape (n, 1)
            mean_function: the approximate mean function of the Gaussian measure
            kernel: the approximate kernel of the Gaussian measure
        """
        super().__init__(x, y, mean_function, kernel)
        self.kernel = kernel

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGaussianMeasureParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return ApproximateGaussianMeasureParameters(
            reference_gaussian_measure=self.kernel.reference_gaussian_measure_parameters,
            mean_function=self.mean_function.generate_parameters(
                parameters["mean_function"]
            ),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximateGaussianMeasureParameters:
        """
        Initialise each parameter of the Gaussian measure with the appropriate random initialisation.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        return ApproximateGaussianMeasureParameters(
            mean_function=self.mean_function.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )

    def compute_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ApproximateGaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)
            parameters: parameters of the Gaussian measure

        Returns: a scalar representing the empirical expected log likelihood

        """
        if not isinstance(parameters, ApproximateGaussianMeasureParameters):
            parameters = self.generate_parameters(parameters)
        return self._compute_expected_log_likelihood(
            mean=self.calculate_mean(x=x, parameters=parameters),
            covariance=self.calculate_covariance(x=x, parameters=parameters),
            observation_noise=jnp.exp(
                self.kernel.reference_gaussian_measure_parameters.log_observation_noise
            ),
            x=x,
            y=y,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_mean(
        self, parameters: ApproximateGaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean by evaluating the mean function at the test points.
            - n is the number of points in x
            - d is the number of dimensions

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            x: design matrix of shape (n, d)
            parameters: parameters of the Gaussian measure

        Returns: the mean function evaluations of shape (n, 1)

        """
        return self.mean_function.predict(x=x, parameters=parameters.mean_function)

    @decorators.preprocess_inputs
    @decorators.check_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_covariance(
        self,
        parameters: ApproximateGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance by evaluating the kernel gram matrix at the test points.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance of shape (n, m)

        """
        # if y is None, compute for x and x
        if y is None:
            y = x

        return self.kernel.calculate_gram(x=x, y=y, parameters=parameters.kernel)
