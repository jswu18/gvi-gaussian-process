from typing import Dict, Union

import pydantic
from flax.core import FrozenDict
from jax import jit
from jax import numpy as jnp

from src.gaussian_measures.gaussian_measures import GaussianMeasure, PRNGKey
from src.gaussian_wasserstein_metric import compute_gaussian_wasserstein_metric
from src.kernels.approximate_kernels import ApproximateKernel
from src.mean_functions.approximate_mean_functions import ApproximateMeanFunction
from src.parameters.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasureParameters,
)
from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.utils.custom_types import JaxFloatType


class ApproximateGaussianMeasure(GaussianMeasure):
    """
    A Gaussian measure which uses an approximate kernel and mean function. These are defined with respect
    to reference Gaussian measures.
    """

    Parameters = ApproximateGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: ApproximateMeanFunction,
        kernel: ApproximateKernel,
        reference_gaussian_measure: GaussianMeasure,
        reference_gaussian_measure_parameters: GaussianMeasureParameters,
        eigenvalue_regularisation: float = 0.0,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
    ):
        """
        Defining the training data (x, y), the mean function, and the kernel for the approximate Gaussian measure.
            - n is the number of training points
            - d is the number of dimensions

        Args:
            x: the training inputs design matrix of shape (n, d)
            y: the training outputs response vector of shape (n, 1)
            mean_function: the approximate mean function of the Gaussian measure
            kernel: the approximate kernel of the Gaussian measure
        """
        super().__init__(x, y, mean_function, kernel)
        self.kernel = kernel

        # define a jit-compiled function to compute the Gaussian Wasserstein metric
        self._jit_compiled_compute_gaussian_wasserstein_metric = jit(
            lambda x_batch, gaussian_measure_parameters_dict: compute_gaussian_wasserstein_metric(
                p=reference_gaussian_measure,
                q=self,
                p_parameters=reference_gaussian_measure_parameters.dict(),
                q_parameters=gaussian_measure_parameters_dict,
                x_batch=x_batch,
                x_train=x,
                eigenvalue_regularisation=eigenvalue_regularisation,
                is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
            )
        )

        # define a jit-compiled function to compute the Gaussian Wasserstein inference loss
        self._jit_compute_gaussian_wasserstein_inference_loss = jit(
            lambda x_batch, gaussian_measure_parameters_dict: (
                self._jit_compiled_compute_gaussian_wasserstein_metric(
                    x_batch, gaussian_measure_parameters_dict
                )
                + self._jit_compiled_compute_negative_expected_log_likelihood(
                    gaussian_measure_parameters_dict
                )
            )
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGaussianMeasureParameters:
        """
        Generates a Pydantic model of the parameters for Approximate Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Approximate Gaussian Measures.

        Returns: A Pydantic model of the parameters for Approximate Gaussian Measures.

        """
        return ApproximateGaussianMeasure.Parameters(
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
        Initialise each parameter of the Approximate Gaussian measure with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Approximate Gaussian Measures.

        """
        return ApproximateGaussianMeasure.Parameters(
            mean_function=self.mean_function.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )

    def _calculate_mean(
        self, parameters: ApproximateGaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean by evaluating the mean function at the test points.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n, 1)

        """
        return self.mean_function.predict(x=x, parameters=parameters.mean_function)

    def _calculate_covariance(
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

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance matrix of shape (n, m)

        """
        return self.kernel.calculate_gram(x=x, y=y, parameters=parameters.kernel)

    def _calculate_observation_noise(
        self, parameters: ApproximateGaussianMeasureParameters = None
    ) -> JaxFloatType:
        return jnp.exp(
            self.kernel.reference_gaussian_measure_parameters.log_observation_noise
        ).astype(float)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_gaussian_wasserstein_metric(
        self,
        parameters: Union[Dict, FrozenDict, ApproximateGaussianMeasureParameters],
        x_batch: jnp.ndarray,
    ) -> float:
        """
        Compute the Gaussian Wasserstein metric between the reference Gaussian measure
        and the approximate Gaussian measure.

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x_batch: a batch of points to compute the dissimilarity measure between the reference gaussian measure and
                     the approximate gaussian measure

        Returns: the Gaussian Wasserstein metric between the reference and approximate Gaussian measures.

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self._jit_compiled_compute_gaussian_wasserstein_metric(
            x_batch, parameters.dict()
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_gaussian_wasserstein_inference_loss(
        self,
        parameters: Union[Dict, FrozenDict, ApproximateGaussianMeasureParameters],
        x_batch: jnp.ndarray,
    ) -> float:
        """
        Compute the Gaussian Wasserstein Inference loss of the approximate gaussian measure is computed as
        the summation of the negative expected log likelihood and the Gaussian Wasserstein metric between the
        reference Gaussian measure and the approximate Gaussian measure.
            - n is the number of points
            - d is the number of dimensions

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x_batch: the design matrix of the batch of training data points of shape (n, d)

        Returns: The loss of the approximate gaussian measure.

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self._jit_compute_gaussian_wasserstein_inference_loss(
            x_batch, parameters.dict()
        )
