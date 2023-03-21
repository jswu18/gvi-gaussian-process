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
        gaussian_measures: List[GaussianMeasure],
        hermite_polynomial_order: int = 50,
    ):
        self.gaussian_measures = gaussian_measures
        self.hermite_roots, self.hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        self._calculate_means = jit(
            lambda x, parameters: vmap(
                lambda gaussian_measure, parameters_: gaussian_measure.calculate_mean(
                    parameters=parameters_, x=x
                )
            )(gaussian_measures, parameters)
        )
        self._calculate_covariances = jit(
            lambda x, parameters: vmap(
                lambda gaussian_measure, parameters_: gaussian_measure.calculate_covariance(
                    parameters=parameters_, x=x
                )
            )(gaussian_measures, parameters)
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self, parameters: Union[Dict, FrozenDict], x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the probabilities of each class
            - n is the number of points in x
            - d is the number of dimensions
            - k is the number of classes

        Args:
            parameters: parameters of the mean function
            x: design matrix of shape (n, d)

        Returns: the mean function evaluated at the points in x of shape (n, k)

        """
        """
        Predicts the class of a point x.
        """
        norm.cdf(x)
        means = self._calculate_means(x=x, parameters=parameters)  # k x n
        covariances = self._calculate_covariances(x=x, parameters=parameters)
        stdev_diagonal = jnp.sqrt(jnp.diagonal(covariances, axis1=1, axis2=2))  # k x n


class ReferenceClassificationModel(ClassificationModel):
    Parameters = ReferenceClassificationModelParameters

    def __init__(
        self,
        gaussian_measures: List[ReferenceGaussianMeasure],
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
        gaussian_measures: List[ApproximateGaussianMeasure],
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
