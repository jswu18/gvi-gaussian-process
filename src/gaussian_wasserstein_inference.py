from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit

from src.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasure,
)
from src.gaussian_measures.gaussian_measures import GaussianMeasure
from src.gaussian_wasserstein_metric import gaussian_wasserstein_metric
from src.module import Module
from src.parameters.gaussian_measures.approximate_gaussian_measure import (
    ApproximateGaussianMeasureParameters,
)
from src.parameters.gaussian_measures.reference_gaussian_measure import (
    ReferenceGaussianMeasureParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class GaussianWassersteinInference(Module):
    """
    A framework for learning the parameters of an approximate gaussian measure by minimising the Gaussian Wasserstein
    distance between the reference gaussian measure and the approximate gaussian measure.
    """

    Parameters = ApproximateGaussianMeasureParameters

    def __init__(
        self,
        reference_gaussian_measure: GaussianMeasure,
        approximate_gaussian_measure: ApproximateGaussianMeasure,
        reference_gaussian_measure_parameters: ReferenceGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
        eigenvalue_regularisation: float = 0.0,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
    ):
        """
        Defining the inference problem.
            - n is the number of training points
            - d is the number of dimensions

        Args:
            reference_gaussian_measure: the reference gaussian measure to approximate
            approximate_gaussian_measure: the approximate gaussian measure to learn
            reference_gaussian_measure_parameters: the parameters of the reference gaussian measure
            x: the design matrix of the training data points of shape (n, d)
            y: the response vector of the training data points of shape (n, 1)
            eigenvalue_regularisation: the regularisation to add to the covariance matrix during eigenvalue computation
            is_eigenvalue_regularisation_absolute_scale: whether the regularisation is an absolute or relative scale
        """
        self.reference_gaussian_measure = reference_gaussian_measure
        self.approximate_gaussian_measure = approximate_gaussian_measure
        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )
        self.x = x
        self.y = y
        self.eigenvalue_regularisation = eigenvalue_regularisation
        self.is_eigenvalue_regularisation_absolute_scale = (
            is_eigenvalue_regularisation_absolute_scale
        )

        # define a jit-compiled function to compute the negative expected log likelihood
        self._jit_compiled_compute_negative_expected_log_likelihood = jit(
            lambda approximate_gaussian_measure_parameters_dict: (
                approximate_gaussian_measure.compute_expected_log_likelihood(
                    x=x,
                    y=y,
                    parameters=approximate_gaussian_measure_parameters_dict,
                )
            )
        )

        # define a jit-compiled function to compute the dissimilarity measure
        self._jit_compiled_compute_dissimilarity_measure = jit(
            lambda x_batch, approximate_gaussian_measure_parameters_dict: gaussian_wasserstein_metric(
                p=reference_gaussian_measure,
                q=approximate_gaussian_measure,
                p_parameters=reference_gaussian_measure_parameters.dict(),
                q_parameters=approximate_gaussian_measure_parameters_dict,
                x_batch=x_batch,
                x_train=x,
                eigenvalue_regularisation=eigenvalue_regularisation,
                is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
            )
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_negative_expected_log_likelihood(
        self,
        approximate_gaussian_measure_parameters: ApproximateGaussianMeasureParameters,
    ) -> float:
        """
        Jit needs a dictionary of parameters to be passed to it to allow for jit compilation.

        Args: approximate_gaussian_measure_parameters: the parameters of the approximate gaussian measure

        Returns: the negative expected log likelihood of the approximate gaussian measure

        """
        return self._jit_compiled_compute_negative_expected_log_likelihood(
            approximate_gaussian_measure_parameters.dict()
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_dissimilarity_measure(
        self,
        x_batch: jnp.ndarray,
        approximate_gaussian_measure_parameters: ApproximateGaussianMeasureParameters,
    ) -> float:
        """
        Jit needs a dictionary of parameters to be passed to it to allow for jit compilation.

        Args:
            x_batch: a batch of points to compute the dissimilarity measure between the reference gaussian measure and
                     the approximate gaussian measure
            approximate_gaussian_measure_parameters: the parameters of the approximate gaussian measure

        Returns: the dissimilarity measure between the reference gaussian measure and the approximate gaussian measure

        """
        return self._jit_compiled_compute_dissimilarity_measure(
            x_batch, approximate_gaussian_measure_parameters.dict()
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ApproximateGaussianMeasureParameters:
        """
        Generates a Pydantic model of the parameters for the Approximate Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Approximate Gaussian Measures.

        Returns: A Pydantic model of the parameters for Approximate Gaussian Measures.

        """
        return self.approximate_gaussian_measure.generate_parameters(parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximateGaussianMeasureParameters:
        """
        The parameters of Gaussian Wasserstein Inference are the Approximate Gaussian Measure Parameters.
        These are initialised using a random key.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Approximate Gaussian Measures.

        """
        return self.approximate_gaussian_measure.initialise_random_parameters(key)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_loss(
        self,
        parameters: Union[Dict, FrozenDict, ApproximateGaussianMeasureParameters],
        x_batch: jnp.ndarray,
    ) -> float:
        """
        Compute the loss of the approximate gaussian measure is computed as the summation of the negative expected
        log likelihood and the dissimilarity measure
            - n is the number of points
            - d is the number of dimensions

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x_batch: the design matrix of the batch of training data points of shape (n, d)

        Returns: The loss of the approximate gaussian measure.

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, ApproximateGaussianMeasureParameters):
            parameters = self.generate_parameters(parameters)

        negative_expected_log_likelihood = (
            self.compute_negative_expected_log_likelihood(
                approximate_gaussian_measure_parameters=parameters
            )
        )
        dissimilarity_measure = self.compute_dissimilarity_measure(
            x_batch=x_batch,
            approximate_gaussian_measure_parameters=parameters,
        )
        return negative_expected_log_likelihood + dissimilarity_measure
