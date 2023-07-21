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
    TemperedClassificationModelParameters,
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
        self._is_jit_negative_expected_log_likelihood_warmed_up = False
        self._jit_compiled_compute_negative_expected_log_likelihood = jit(
            lambda parameters_dict, x, y: (
                self._compute_negative_log_likelihood(
                    x=x,
                    y=y,
                    parameters=parameters_dict,
                )
            )
        )

        self._is_jit_predict_probability_warmed_up = False
        self._jit_compiled_predict_probability = jit(
            lambda parameters_dict, x: (
                self._predict_probability(
                    x=x,
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

    @abstractmethod
    def _calculate_covariances(
        self,
        parameters: ClassificationModelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        full_cov: bool = True,
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
            full_cov: whether to calculate the full covariance matrices

        Returns: covariances of the Gaussian measures for the classification model of shape (k, n, m) or (k, n)

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
        full_cov: bool = True,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        return self._calculate_covariances(
            x=x, y=y, parameters=parameters, full_cov=full_cov
        )

    @staticmethod
    def calculate_s_matrix(
        means: jnp.ndarray,
        covariance_diagonals: jnp.ndarray,
        hermite_weights: jnp.ndarray,
        hermite_roots: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the probabilities of each class
            - n is the number of points in x
            - d is the number of dimensions
            - k is the number of classes
            - h is the number of Hermite roots and weights

        Args:
            means: means of the Gaussian measures for the classification model of shape (k, n)
            covariance_diagonals: covariances diagonal of the Gaussian measures for the classification model of shape (k, n)
            hermite_weights: weights of the Hermite polynomials of shape (h,)
            hermite_roots: roots of the Hermite polynomials of shape (h,)

        Returns: the mean function evaluated at the points in x of shape (n, k)

        """
        """
        Predicts the class of a point x.
        """

        # (k, n)
        stdev_diagonals = jnp.sqrt(covariance_diagonals)

        # (h, k_j, k_l, n) where k_j = k_l = k, used to keep track of the indices
        cdf_input = jnp.divide(
            (
                # (h, None, None, None), (None, k_j, None, n) -> (h, k_j, k_l, n)
                jnp.sqrt(2)
                * jnp.multiply(
                    hermite_roots[:, None, None, None],
                    stdev_diagonals[None, :, None, :],
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
            stdev_diagonals[
                None,
                None,
                :,
            ],
        )

        log_cdf_values = jnp.log(norm.cdf(cdf_input))  # (h, k_j, k_l, n)

        # (h, k, n)
        hermite_components = jnp.multiply(
            # (h, None, None) -> (h, k_j, n)
            hermite_weights[:, None, None],
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

    def _compute_negative_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        # (n, k)
        s_matrix = self.calculate_s_matrix(
            means=self._calculate_means(parameters=parameters, x=x),
            covariance_diagonals=self._calculate_covariances(
                parameters=parameters, x=x, full_cov=False
            ),
            hermite_weights=self._hermite_weights,
            hermite_roots=self._hermite_roots,
        )

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
    def compute_negative_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        if not self._is_jit_negative_expected_log_likelihood_warmed_up:
            self._jit_compiled_compute_negative_expected_log_likelihood(
                parameters.dict(), x[:2, ...], y[:2, ...]
            )
            self._is_jit_negative_expected_log_likelihood_warmed_up = True
        return self._jit_compiled_compute_negative_expected_log_likelihood(
            parameters.dict(),
            x,
            y,
        )

    def _predict_probability(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        # (n, k)
        s_matrix = self.calculate_s_matrix(
            means=self._calculate_means(parameters=parameters, x=x),
            covariance_diagonals=self._calculate_covariances(
                parameters=parameters, x=x, full_cov=False
            ),
            hermite_weights=self._hermite_weights,
            hermite_roots=self._hermite_roots,
        )

        # (n, k)
        return (1 - self.epsilon) * s_matrix + (
            self.epsilon / (self.number_of_labels - 1)
        ) * (1 - s_matrix)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_probability(
        self,
        parameters: Union[Dict, FrozenDict, ClassificationModelParameters],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        if not self._is_jit_predict_probability_warmed_up:
            self._jit_compiled_predict_probability(parameters.dict(), x[:2, ...])
            self._is_jit_predict_probability_warmed_up = True
        return self._jit_compiled_predict_probability(parameters.dict(), x)


class TemperedClassificationModel(ClassificationModel):
    Parameters = TemperedClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        classification_model: ClassificationModel,
        classification_model_parameters: ClassificationModelParameters,
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x,
            y=y,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
        )
        self.classification_model = classification_model
        self.classification_model_parameters = classification_model_parameters

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> TemperedClassificationModelParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        return self.Parameters(
            log_tempering_factors=parameters["log_tempering_factors"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> TemperedClassificationModelParameters:
        """
        Initialise each parameter of the Module with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for the Module.

        """
        pass

    @property
    def labels(self) -> List[Any]:
        """
        Returns a list of the labels for the classification model.
        Returns: A list of the labels for the classification model.

        """
        return self.classification_model.labels

    @property
    def number_of_labels(self) -> int:
        """
        Returns the number of labels for the classification model.
        Returns: The number of labels for the classification model.

        """
        return self.classification_model.number_of_labels

    def _calculate_covariances(
        self,
        parameters: TemperedClassificationModelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        full_cov=False,
    ):
        if full_cov:
            return jnp.multiply(
                jnp.exp(parameters.log_tempering_factors)[:, None, None],
                self.classification_model.calculate_covariances(
                    x=x,
                    y=y,
                    parameters=self.classification_model_parameters,
                    full_cov=full_cov,
                ),
            )
        else:
            return jnp.multiply(
                jnp.exp(parameters.log_tempering_factors)[:, None],
                self.classification_model.calculate_covariances(
                    x=x,
                    y=y,
                    parameters=self.classification_model_parameters,
                    full_cov=full_cov,
                ),
            )

    def _calculate_means(
        self,
        parameters: TemperedClassificationModelParameters,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.classification_model.calculate_means(
            x=x, parameters=self.classification_model_parameters
        )
