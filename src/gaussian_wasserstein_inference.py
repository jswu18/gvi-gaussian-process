from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit

from src.gaussian_measures.approximate_gaussian_measure import (
    ApproximateGaussianMeasure,
)
from src.gaussian_measures.gaussian_measure import GaussianMeasure
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
        self.compute_negative_expected_log_likelihood = jit(
            lambda approximate_gaussian_measure_parameters_dict: (
                approximate_gaussian_measure.compute_expected_log_likelihood(
                    x=x,
                    y=y,
                    parameters=approximate_gaussian_measure_parameters_dict,
                )
            )
        )
        self.compute_dissimilarity_measure = jit(
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
    def generate_parameters(
        self, parameters: FrozenDict
    ) -> ApproximateGaussianMeasureParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return self.approximate_gaussian_measure.Parameters(**parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximateGaussianMeasureParameters:
        """
        Randomly initialise the parameters of the approximate gaussian measure by calling the
        initialise_random_parameters method of the approximate gaussian measure and passing the given parameters.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the approximate gaussian measure.

        """
        return self.approximate_gaussian_measure.initialise_random_parameters(key)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_loss(
        self,
        parameters_dict: Union[Dict, FrozenDict],
        x_batch: jnp.ndarray,
    ) -> float:
        """
        Compute the loss of the approximate gaussian measure by adding the negative expected log likelihood and the
        dissimilarity measure.
            - n is the number of points
            - d is the number of dimensions

        Args:
            parameters_dict: the parameters of the approximate gaussian measure
            x_batch: the design matrix of the batch of training data points of shape (n, d)

        Returns: The loss of the approximate gaussian measure.

        """
        negative_expected_log_likelihood = (
            self.compute_negative_expected_log_likelihood(
                approximate_gaussian_measure_parameters_dict=parameters_dict
            )
        )
        dissimilarity_measure = self.compute_dissimilarity_measure(
            x_batch=x_batch,
            approximate_gaussian_measure_parameters_dict=parameters_dict,
        )
        return negative_expected_log_likelihood + dissimilarity_measure
