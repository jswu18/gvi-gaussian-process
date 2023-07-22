from abc import ABC
from typing import Dict, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax.core.frozen_dict import FrozenDict
from scipy.special import roots_hermite

from src.distributions import Multinomial
from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels.multi_output_kernel import (
    MultiOutputKernel,
    MultiOutputKernelParameters,
)
from src.means.base import MeanBase


class GPClassificationBaseParameters(GPBaseParameters):
    kernel: MultiOutputKernelParameters


class GPClassificationBase(GPBase, ABC):

    Parameters = GPClassificationBaseParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: MultiOutputKernel,
        epsilon: float,
        hermite_polynomial_order: int,
    ):
        assert mean.number_output_dimensions > 1
        assert mean.number_output_dimensions == kernel.number_output_dimensions
        self.number_output_dimensions = mean.number_output_dimensions
        self.epsilon = epsilon
        self._hermite_roots, self._hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        GPBase.__init__(self, mean=mean, kernel=kernel)

    def _construct_distribution(self, probabilities: jnp.ndarray):
        return Multinomial(probabilities=probabilities)

    def _predict_probability(
        self,
        parameters: Union[Dict, FrozenDict, GPClassificationBaseParameters],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        mean, covariance_diagonals = self._calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=False,
        )
        s_matrix = self._calculate_s_matrix(
            means=mean,
            covariance_diagonals=covariance_diagonals,
            hermite_weights=self._hermite_weights,
            hermite_roots=self._hermite_roots,
        )
        # (n, k)
        probabilities = (1 - self.epsilon) * s_matrix + (
            self.epsilon / (self.number_output_dimensions - 1)
        ) * (1 - s_matrix)
        return probabilities

    @staticmethod
    def _calculate_s_matrix(
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

        log_cdf_values = jnp.log(jsp.stats.norm.cdf(cdf_input))  # (h, k_j, k_l, n)

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