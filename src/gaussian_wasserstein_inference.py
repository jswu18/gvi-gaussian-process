from typing import Any

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit

from src.gaussian_measures import GaussianMeasure
from src.gaussian_wasserstein_metric import gaussian_wasserstein_metric
from src.module import Module

PRNGKey = Any  # pylint: disable=invalid-name


class GaussianWassersteinInference(Module):
    def __init__(
        self,
        reference_gaussian_measure: GaussianMeasure,
        approximation_gaussian_measure: GaussianMeasure,
        reference_gaussian_measure_parameters: FrozenDict,
        x: jnp.ndarray,
        y: jnp.ndarray,
        eigenvalue_regularisation: float = 0.0,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
    ):
        self.reference_gaussian_measure = reference_gaussian_measure
        self.approximation_gaussian_measure = approximation_gaussian_measure
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
            lambda approximation_gaussian_measure_parameters: (
                approximation_gaussian_measure.compute_expected_log_likelihood(
                    x=x,
                    y=y,
                    observation_noise=jnp.exp(
                        reference_gaussian_measure_parameters["log_observation_noise"]
                    ),
                    parameters=approximation_gaussian_measure_parameters,
                )
            )
        )
        self.compute_dissimilarity_measure = jit(
            lambda x_batch, approximation_gaussian_measure_parameters: gaussian_wasserstein_metric(
                p=reference_gaussian_measure,
                q=approximation_gaussian_measure,
                p_parameters=reference_gaussian_measure_parameters,
                q_parameters=approximation_gaussian_measure_parameters,
                x_batch=x_batch,
                x_train=x,
                eigenvalue_regularisation=eigenvalue_regularisation,
                is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
            )
        )

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        return self.approximation_gaussian_measure.initialise_random_parameters(key)

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        return self.approximation_gaussian_measure.initialise_parameters(**kwargs)

    def compute_loss(
        self,
        approximation_gaussian_measure_parameters: FrozenDict,
        x_batch: jnp.ndarray,
    ) -> float:
        negative_expected_log_likelihood = (
            self.compute_negative_expected_log_likelihood(
                approximation_gaussian_measure_parameters
            )
        )
        dissimilarity_measure = self.compute_dissimilarity_measure(
            x_batch=x_batch,
            approximation_gaussian_measure_parameters=approximation_gaussian_measure_parameters,
        )
        return negative_expected_log_likelihood + dissimilarity_measure
