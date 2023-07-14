from abc import ABC
from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit

from src.classification_models.classification_models import ClassificationModel
from src.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasure,
)
from src.gaussian_measures.gaussian_measures import GaussianMeasure
from src.gaussian_measures.reference_gaussian_measures import ReferenceGaussianMeasure
from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)
from src.parameters.classification_models.distinct_means import (
    ApproximateDistinctMeansClassificationModelParameters,
    DistinctMeansClassificationModelParameters,
    ReferenceDistinctMeansClassificationModelParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class DistinctMeansClassificationModel(ClassificationModel, ABC):
    Parameters = DistinctMeansClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, GaussianMeasure],
        epsilon: float,
        hermite_polynomial_order: int,
    ):
        super().__init__(
            x=x, y=y, epsilon=epsilon, hermite_polynomial_order=hermite_polynomial_order
        )
        self.gaussian_measures = gaussian_measures

    @property
    def labels(self) -> List[Any]:
        return sorted(list(self.gaussian_measures.keys()))

    @property
    def number_of_labels(self) -> int:
        return len(self.gaussian_measures)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> DistinctMeansClassificationModelParameters:
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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> DistinctMeansClassificationModelParameters:
        """
        Initialise each parameter of the Module with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for the Module.

        """
        pass

    def _calculate_means(
        self,
        x: jnp.ndarray,
        parameters: DistinctMeansClassificationModelParameters,
    ) -> jnp.ndarray:
        """
        Calculates the means of the Gaussian measures for the classification model
        by evaluating the mean function of each distinct Gaussian measure.
            - k is the number of classes
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            parameters: parameters of the classification model

        Returns: means of the Gaussian measures for the classification model of shape (k, n)

        """
        return jnp.array(
            [
                self.gaussian_measures[label].calculate_mean(
                    parameters=parameters.gaussian_measures[label], x=x
                )
                for label in self.labels
            ]
        )

    def _calculate_covariances(
        self,
        parameters: DistinctMeansClassificationModelParameters,
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
        return jnp.array(
            [
                self.gaussian_measures[label].calculate_covariance(
                    parameters=parameters.gaussian_measures[label],
                    x=x,
                    y=y,
                    full_cov=full_cov,
                )
                for label in self.labels
            ]
        )


class ReferenceDistinctMeansClassificationModel(DistinctMeansClassificationModel):
    Parameters = ReferenceDistinctMeansClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, ReferenceGaussianMeasure],
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x,
            y=y,
            gaussian_measures=gaussian_measures,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
        )


class ApproximateDistinctMeansClassificationModel(DistinctMeansClassificationModel):
    Parameters = ApproximateDistinctMeansClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measures: Dict[Any, ApproximateGaussianMeasure],
        reference_classification_model: ClassificationModel,
        reference_classification_model_parameters: ClassificationModelParameters,
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x,
            y=y,
            gaussian_measures=gaussian_measures,
            epsilon=epsilon,
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
                    parameters_dict=parameters_dict,
                    x=self.x,
                    y=self.y,
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
    def compute_gaussian_wasserstein_inference_loss(
        self,
        parameters: Union[
            Dict, FrozenDict, ApproximateDistinctMeansClassificationModelParameters
        ],
        x_batch: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self._jit_compiled_compute_gaussian_wasserstein_inference_loss(
            parameters_dict=parameters.dict(), x_batch=x_batch
        )
