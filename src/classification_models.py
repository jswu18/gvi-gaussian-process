from abc import ABC
from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit, vmap
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
        gaussian_measures: Dict[Any, GaussianMeasure],
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        self.gaussian_measures = gaussian_measures
        self.epsilon = epsilon
        self._hermite_roots, self._hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        self._calculate_means = jit(
            lambda x, parameters: vmap(
                lambda gaussian_measure, parameters_: gaussian_measure.calculate_mean(
                    parameters=parameters_, x=x
                )
            )(self.gaussian_measures, parameters)
        )
        self._calculate_covariances = jit(
            lambda x, parameters: vmap(
                lambda gaussian_measure, parameters_: gaussian_measure.calculate_covariance(
                    parameters=parameters_, x=x
                )
            )(self.gaussian_measures, parameters)
        )

    @property
    def labels(self) -> List[Any]:
        return list(self.gaussian_measures.keys())

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
        # means = self._calculate_means(x=x, parameters=parameters.gaussian_measures.values())  # k x n
        # covariances = self._calculate_covariances(x=x, parameters=parameters.gaussian_measures.values())

        # ugly and slow hack for now
        means = []
        covariances = []
        for i, label in enumerate(self.labels):
            means.append(
                self.gaussian_measures[label].calculate_mean(
                    x=x, parameters=parameters.gaussian_measures[label]
                )
            )
            covariances.append(
                self.gaussian_measures[label].calculate_covariance(
                    x=x, parameters=parameters.gaussian_measures[label]
                )
            )
        means = jnp.array(means)
        covariances = jnp.array(covariances)

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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_negative_log_likelihood(
        self, parameters: ClassificationModelParameters, x: jnp.ndarray, y: jnp.ndarray
    ) -> float:
        # (n, k)
        s_matrix = self._calculate_s_matrix(parameters=parameters, x=x)

        # (n, k)
        return -jnp.multiply(
            jnp.log(1 - self.epsilon) * s_matrix
            + jnp.log(self.epsilon / (self.number_of_labels - 1)) * (1 - s_matrix),
            y,  # to mask out the other classes
        ).sum()

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


class ReferenceClassificationModel(ClassificationModel):
    Parameters = ReferenceClassificationModelParameters

    def __init__(
        self,
        gaussian_measures: Dict[Any, ReferenceGaussianMeasure],
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            gaussian_measures=gaussian_measures,
            hermite_polynomial_order=hermite_polynomial_order,
        )

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
        pass

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


class ApproximateClassificationModel(Module):
    Parameters = ApproximateClassificationModelParameters

    def __init__(
        self,
        gaussian_measures: Dict[Any, ApproximateGaussianMeasure],
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            gaussian_measures=gaussian_measures,
            hermite_polynomial_order=hermite_polynomial_order,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ApproximateClassificationModelParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        pass

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
