from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.stats import norm
from scipy.special import roots_hermite

from src.module import Module
from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class ClassificationModel(Module, ABC):
    """
    A base class for classification models.
    """

    Parameters = ClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        epsilon: float,
        hermite_polynomial_order: int,
    ):
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self._hermite_roots, self._hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        self._jit_compiled_calculate_means = jit(
            lambda x, parameters_dict: self.calculate_means(
                x, parameters_dict["gaussian_measures"]
            )
        )
        self._jit_compiled_calculate_covariances = jit(
            lambda x, parameters_dict: self.calculate_covariances(
                x, parameters_dict["gaussian_measures"]
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
    @abstractmethod
    def labels(self) -> List[Any]:
        """
        Returns a list of the labels for the classification model.
        Returns: A list of the labels for the classification model.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_labels(self) -> int:
        """
        Returns the number of labels for the classification model.
        Returns: The number of labels for the classification model.

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_means(
        self,
        parameters: ClassificationModelParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculates the means of the Gaussian measures for the classification model.
            - k is the number of classes
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the classification model
            x: design matrix of shape (n, d)

        Returns: means of the Gaussian measures for the classification model of shape (k, n)

        """
        raise NotImplementedError

    def _calculate_covariances(
        self,
        parameters: ClassificationModelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculates the covariances of the Gaussian measures for the classification model.
            - k is the number of classes
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the classification model
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: covariances of the Gaussian measures for the classification model of shape (k, n, m)

        """
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_means(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        return self._calculate_means(x=x, parameters=parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_covariances(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        return self._calculate_covariances(x=x, y=y, parameters=parameters)

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
        means = self._calculate_means(x=x, parameters=parameters)  # k x n
        covariances = self._calculate_covariances(x=x, parameters=parameters)

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
