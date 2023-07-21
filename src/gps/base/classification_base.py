from abc import ABC

import jax.numpy as jnp
import jax.scipy as jsp
from scipy.special import roots_hermite

from src.distributions import Multinomial
from src.gaussian_processes.base.base import GaussianProcessBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.parameters.gaussian_processes.base import GaussianProcessBaseParameters


class GaussianProcessClassificationBase(ABC, GaussianProcessBase):
    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
        epsilon: float,
        hermite_polynomial_order: int,
    ):
        self.epsilon = epsilon
        self._hermite_roots, self._hermite_weights = roots_hermite(
            hermite_polynomial_order
        )
        super().__init__(mean=mean, kernel=kernel)

    def _predict_probability(
        self, parameters: GaussianProcessBaseParameters, x: jnp.ndarray
    ) -> Multinomial:
        gaussian_distribution = self._calculate_prediction_distribution(
            parameters=parameters, x=x, full_coviarance=False
        )
        s_matrix = self._calculate_s_matrix(
            means=gaussian_distribution.mean,
            covariance_diagonals=gaussian_distribution.covariance,
            hermite_weights=self._hermite_weights,
            hermite_roots=self._hermite_roots,
        )
        # (n, k)
        probabilities = (1 - self.epsilon) * s_matrix + (
            self.epsilon / (self.number_of_labels - 1)
        ) * (1 - s_matrix)
        return Multinomial(probabilities=probabilities)

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

        log_cdf_values = jnp.log(
            jsp.scipy.stats.norm.cdf(cdf_input)
        )  # (h, k_j, k_l, n)

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
