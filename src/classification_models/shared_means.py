from typing import Any, Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit, vmap

from src.classification_models.classification_models import ClassificationModel
from src.gaussian_wasserstein_metric import (
    compute_gaussian_wasserstein_metric_from_grams,
)
from src.kernels.kernels import Kernel
from src.mean_functions.classification_shared_mean_functions import SharedMeanFunction
from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)
from src.parameters.classification_models.shared_means import (
    SharedMeansClassificationModelParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class SharedMeansClassificationModel(ClassificationModel):
    Parameters = SharedMeansClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: SharedMeanFunction,
        kernels: Dict[Any, Kernel],
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
    ):
        super().__init__(
            x=x, y=y, epsilon=epsilon, hermite_polynomial_order=hermite_polynomial_order
        )
        self.mean_function = mean_function
        self.kernels = kernels

    @property
    def labels(self) -> List[Any]:
        return sorted(list(self.kernels.keys()))

    @property
    def number_of_labels(self) -> int:
        return len(self.kernels)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> SharedMeansClassificationModelParameters:
        """
        Generates a Pydantic model of the parameters for the Module.

        Args:
            parameters: A dictionary of the parameters for the Module.

        Returns: A Pydantic model of the parameters for the Module.

        """
        return self.Parameters(
            mean_function=self.mean_function.generate_parameters(
                parameters["mean_function"]
            ),
            kernels={
                label: self.kernels[label].generate_parameters(
                    parameters["kernels"][label]
                )
                for label in self.labels
            },
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> SharedMeansClassificationModelParameters:
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
        parameters: SharedMeansClassificationModelParameters,
    ) -> jnp.ndarray:
        """
        Calculates the means of the Gaussian measures for the classification model
        by evaluating the shared mean function.
            - k is the number of classes
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            parameters: parameters of the classification model

        Returns: means of the the classification model of shape (k, n)

        """
        return self.mean_function.predict(x=x, parameters=parameters.mean_function)

    def _calculate_covariances(
        self,
        parameters: SharedMeansClassificationModelParameters,
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
        return jnp.array(
            [
                self.kernels[label].calculate_gram(
                    parameters=parameters.kernels[label],
                    x=x,
                    y=y,
                )
                for label in self.labels
            ]
        )


class ApproximateSharedMeansClassificationModel(SharedMeansClassificationModel):
    Parameters = SharedMeansClassificationModelParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: SharedMeanFunction,
        kernels: Dict[Any, Kernel],
        reference_classification_model: ClassificationModel,
        reference_classification_model_parameters: ClassificationModelParameters,
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
        eigenvalue_regularisation: float = 0.0,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
        use_symmetric_matrix_eigendecomposition: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            mean_function=mean_function,
            kernels=kernels,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
        )
        self.reference_classification_model = reference_classification_model
        self.reference_classification_model_parameters = (
            reference_classification_model_parameters
        )
        self._jit_compiled_compute_gaussian_wasserstein_inference_loss = jit(
            lambda parameters_dict, x_batch: (
                self._jit_compiled_compute_negative_expected_log_likelihood(
                    parameters_dict=parameters_dict
                )
                + jnp.sum(
                    vmap(
                        lambda m_p, c_p, m_q, c_q, c_bt_p, c_bt_q: compute_gaussian_wasserstein_metric_from_grams(
                            mean_train_p=m_p,
                            covariance_train_p=c_p,
                            mean_train_q=m_q,
                            covariance_train_q=c_q,
                            gram_batch_train_p=c_bt_p,
                            gram_batch_train_q=c_bt_q,
                            eigenvalue_regularisation=eigenvalue_regularisation,
                            is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
                            use_symmetric_matrix_eigendecomposition=use_symmetric_matrix_eigendecomposition,
                        )
                    )(
                        reference_classification_model.calculate_means(
                            x=x,
                            parameters=reference_classification_model_parameters,
                        ),
                        reference_classification_model.calculate_covariances(
                            x=x,
                            parameters=reference_classification_model_parameters,
                        ),
                        self.calculate_means(
                            x=x,
                            parameters=parameters_dict,
                        ),
                        self.calculate_covariances(
                            x=x,
                            parameters=parameters_dict,
                        ),
                        reference_classification_model.calculate_covariances(
                            x=x_batch,
                            y=x,
                            parameters=reference_classification_model_parameters,
                        ),
                        self.calculate_covariances(
                            x=x_batch,
                            y=x,
                            parameters=parameters_dict,
                        ),
                    )
                )
            )
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_gaussian_wasserstein_inference_loss(
        self,
        parameters: Union[Dict, FrozenDict, SharedMeansClassificationModelParameters],
        x_batch: jnp.ndarray,
    ) -> float:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self._jit_compiled_compute_gaussian_wasserstein_inference_loss(
            parameters_dict=parameters.dict(), x_batch=x_batch
        )
