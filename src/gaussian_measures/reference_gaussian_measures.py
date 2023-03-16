from typing import Dict, Union

import pydantic
from flax.core import FrozenDict
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve

from src.gaussian_measures.gaussian_measures import GaussianMeasure, PRNGKey
from src.kernels.reference_kernels import StandardKernel
from src.mean_functions.mean_functions import MeanFunction
from src.parameters.gaussian_measures.reference_gaussian_measures import (
    ReferenceGaussianMeasureParameters,
)
from src.utils.custom_types import JaxFloatType


class ReferenceGaussianMeasure(GaussianMeasure):
    """
    A reference Gaussian measure defined with respect to a mean function and a kernel.
    """

    Parameters = ReferenceGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: StandardKernel,
    ):
        """
        Defining the training data (x, y), the mean function, and the kernel for the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: the training inputs design matrix of shape (n, d)
            y: the training outputs response vector of shape (n, 1)
            mean_function: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        super().__init__(x, y, mean_function, kernel)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ReferenceGaussianMeasureParameters:
        """
        Generates a Pydantic model of the parameters for Reference Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Reference Gaussian Measures.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        return ReferenceGaussianMeasure.Parameters(
            log_observation_noise=parameters["log_observation_noise"],
            mean_function=self.mean_function.generate_parameters(
                parameters["mean_function"]
            ),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ReferenceGaussianMeasureParameters:
        """
        Initialise each parameter of the Reference Gaussian measure with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        return ReferenceGaussianMeasure.Parameters(
            log_observation_noise=random.normal(key),
            mean_function=self.mean_function.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )

    def _calculate_mean(
        self, parameters: ReferenceGaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean using the formula for the posterior mean of a Gaussian process which is
        m(x) = k(x, X) @ (K(X, X) + σ^2I)^-1 @ y. This is added to the mean function prediction to generate the
        full posterior mean.

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n, 1)

        """
        gram_train = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x=self.x,
        )
        gram_train_test = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x=self.x,
            y=x,
        )
        observation_noise = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = cho_factor(gram_train + observation_noise)
        kernel_mean = gram_train_test.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=self.y
        )
        return kernel_mean + self.mean_function.predict(
            x=x, parameters=parameters.mean_function
        )

    def _calculate_covariance(
        self,
        parameters: ReferenceGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance using the formula for the posterior covariance of a Gaussian process which is
        k(x, y) = k(x, y) - k(x, X) @ (K(X, X) + σ^2I)^-1 @ k(X, y).
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance matrix of shape (n, m)

        """
        gram_train = self.kernel.calculate_gram(x=self.x, parameters=parameters.kernel)
        gram_train_x = self.kernel.calculate_gram(
            x=self.x, y=x, parameters=parameters.kernel
        )
        gram_train_y = self.kernel.calculate_gram(
            x=self.x, y=y, parameters=parameters.kernel
        )
        gram_xy = self.kernel.calculate_gram(x=x, y=y, parameters=parameters.kernel)
        observation_noise_matrix = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = cho_factor(
            gram_train + observation_noise_matrix
        )

        return gram_xy - gram_train_x.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_y
        )

    def _calculate_observation_noise(
        self, parameters: ReferenceGaussianMeasureParameters = None
    ) -> JaxFloatType:
        return jnp.exp(parameters.log_observation_noise).astype(jnp.float64)

    def _compute_negative_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ReferenceGaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: a scalar representing the empirical expected log likelihood

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        mean = self.calculate_mean(x=x, parameters=parameters)
        covariance = self.calculate_covariance(x=x, parameters=parameters)
        observation_noise = self.calculate_observation_noise(parameters=parameters)

        diagonal_covariance = jnp.diag(covariance) + observation_noise
        error = y - mean

        return (x.shape[0] / 2) * (
            jnp.log(2 * jnp.pi)
            + jnp.sum(jnp.log(diagonal_covariance))
            + error.T @ jnp.diag(1 / diagonal_covariance) @ error
        )
