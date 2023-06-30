from abc import ABC
from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.stats import norm
from scipy.special import roots_hermite

from src.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasure,
)
from src.gaussian_measures.gaussian_measures import GaussianMeasure
from src.gaussian_measures.reference_gaussian_measures import ReferenceGaussianMeasure
from src.module import Module
from src.parameters.classification_models import (
    ApproximateClassificationModelParameters,
    ClassificationModelParameters,
    ReferenceClassificationModelParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class ClassificationModel(Module, ABC):
    Parameters = ClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, GaussianMeasure],
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        self.x = x
        self.y = y
        self.gaussian_measures = gaussian_measures
        self.epsilon = epsilon
        self._hermite_roots, self._hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        self._calculate_means = jit(
            lambda x, parameters_dict: jnp.array(
                [
                    self.gaussian_measures[label].calculate_mean(
                        x=x, parameters=parameters_dict["gaussian_measures"][label]
                    )
                    for label in self.labels
                ]
            )
        )
        self._calculate_covariances = jit(
            lambda x, parameters_dict: jnp.array(
                [
                    self.gaussian_measures[label].calculate_covariance(
                        x=x, parameters=parameters_dict["gaussian_measures"][label]
                    )
                    for label in self.labels
                ]
            )
        )
        # define a jit-compiled function to compute the negative expected log likelihood
        self._jit_compiled_compute_negative_expected_log_likelihood = jit(
            lambda parameters_dict: (
                self.compute_negative_log_likelihood(
                    x=x,
                    y=y,
                    parameters=parameters_dict,
                )
            )
        )

    @property
    def labels(self) -> List[Any]:
        return sorted(list(self.gaussian_measures.keys()))

    @property
    def number_of_labels(self) -> int:
        return len(self.gaussian_measures)

    def _calculate_s_matrix(
        self, parameters: ClassificationModelParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the probabilities of each class
            - n is the number of points in x
            - d is the number of dimensions
            - k is the number of classes
            - h is the number of Hermite roots and weights

        Args:
            parameters: parameters of the classification model
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, k)

        """
        """
        Predicts the class of a point x.
        """
        means = self._calculate_means(x=x, parameters_dict=parameters.dict())  # k x n
        covariances = self._calculate_covariances(
            x=x, parameters_dict=parameters.dict()
        )

        # (k, n)
        stdev_diagonal = jnp.sqrt(jnp.diagonal(covariances, axis1=1, axis2=2))

        # (h, k_j, k_l, n) where k_j = k_l = k, used to keep track of the indices
        cdf_input = jnp.divide(
            (
                # (h, None, None, None), (None, k_j, None, n) -> (h, k_j, k_l, n)
                jnp.sqrt(2)
                * jnp.multiply(
                    self._hermite_roots[:, None, None, None],
                    stdev_diagonal[None, :, None, :],
                )
                # (None, k_j, None, n) -> (h, k_j, k_l, n)
                + means[None, :, None, :]
                # (None, None, k_l, n) -> (h, k_j, k_l, n)
                - means[
                    None,
                    None,
                    :,
                ]
            ),
            # (None, None, k_l, n) -> (h, k_j, k_l, n)
            stdev_diagonal[
                None,
                None,
                :,
            ],
        )

        log_cdf_values = jnp.log(norm.cdf(cdf_input))  # (h, k_j, k_l, n)

        # (h, k, n)
        hermite_components = jnp.multiply(
            # (h, None, None) -> (h, k_j, n)
            self._hermite_weights[:, None, None],
            # (h, k_j, n)
            jnp.exp(
                # (h, k_j, k_l, n) -> (h, k_j, n)
                jnp.sum(log_cdf_values, axis=2)
                # remove over counting when j == l for j,l in {1, ..., k}
                # (h, k_j, k_l, n) -> (h, n, k_j) -> (h, k_j, n)
                - jnp.swapaxes(
                    jnp.diagonal(log_cdf_values, axis1=1, axis2=2), axis1=1, axis2=2
                )
            ),
        )

        # (h, k, n) -> (k, n) -> (n, k)
        return ((1 / jnp.sqrt(jnp.pi)) * jnp.sum(hermite_components, axis=0)).T

    # @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_negative_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)

        # (n, k)
        s_matrix = self._calculate_s_matrix(parameters=parameters, x=x)

        # (n, k)
        return (
            -jnp.multiply(
                jnp.log(1 - self.epsilon) * s_matrix
                + jnp.log(self.epsilon / (self.number_of_labels - 1)) * (1 - s_matrix),
                y,  # to mask out the other classes
            )
            .sum(axis=1)
            .mean()
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_probability(
        self, parameters: ClassificationModelParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        # (n, k)
        s_matrix = self._calculate_s_matrix(parameters=parameters, x=x)

        # (n, k)
        return (1 - self.epsilon) * s_matrix + (
            self.epsilon / (self.number_of_labels - 1)
        ) * (1 - s_matrix)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ClassificationModelParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        return self.Parameters(
            gaussian_measures={
                label: self.gaussian_measures[label].generate_parameters(
                    parameters["gaussian_measures"][label]
                )
                for label in self.labels
            }
        )


class ReferenceClassificationModel(ClassificationModel):
    Parameters = ReferenceClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, ReferenceGaussianMeasure],
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x,
            y=y,
            gaussian_measures=gaussian_measures,
            hermite_polynomial_order=hermite_polynomial_order,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ClassificationModelParameters:
        """
        Initialise each parameter of the Module with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for the Module.

        """
        pass


class ApproximateClassificationModel(ClassificationModel):
    Parameters = ApproximateClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, ApproximateGaussianMeasure],
        reference_classification_model: ReferenceClassificationModel,
        reference_classification_model_parameters: ReferenceClassificationModelParameters,
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x,
            y=y,
            gaussian_measures=gaussian_measures,
            hermite_polynomial_order=hermite_polynomial_order,
        )
        self.gaussian_measures = gaussian_measures
        self.reference_classification_model = reference_classification_model
        self.reference_classification_model_parameters = (
            reference_classification_model_parameters
        )
        self._jit_compiled_compute_gaussian_wasserstein_inference_loss = jit(
            lambda parameters_dict, x_batch: (
                self._jit_compiled_compute_negative_expected_log_likelihood(
                    parameters_dict=parameters_dict
                )
                + sum(
                    [
                        self.gaussian_measures[
                            label
                        ].compute_gaussian_wasserstein_metric(
                            parameters=parameters_dict["gaussian_measures"][label],
                            x_batch=x_batch,
                        )
                        for label in self.labels
                    ]
                )
            )
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximateClassificationModelParameters:
        """
        Initialise each parameter of the Module with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for the Module.

        """
        pass

    # @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_gaussian_wasserstein_inference_loss(
        self,
        parameters: Union[Dict, FrozenDict, ApproximateClassificationModelParameters],
        x_batch: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self._jit_compiled_compute_gaussian_wasserstein_inference_loss(
            parameters_dict=parameters.dict(), x_batch=x_batch
        )
