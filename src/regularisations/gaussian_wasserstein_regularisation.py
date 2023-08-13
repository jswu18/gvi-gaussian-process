import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import pydantic

from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase
from src.utils.matrix_operations import (
    add_diagonal_regulariser,
    compute_covariance_eigenvalues,
    compute_product_eigenvalues,
)


class GaussianWassersteinRegularisation(RegularisationBase):
    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        eigenvalue_regularisation: float = 1e-8,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
        use_symmetric_matrix_eigendecomposition: bool = True,
        include_eigendecomposition: bool = False,
    ):
        self.eigenvalue_regularisation = eigenvalue_regularisation
        self.is_eigenvalue_regularisation_absolute_scale = (
            is_eigenvalue_regularisation_absolute_scale
        )
        self.use_symmetric_matrix_eigendecomposition = (
            use_symmetric_matrix_eigendecomposition
        )
        self.include_eigendecomposition = include_eigendecomposition
        super().__init__(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )

    @staticmethod
    def _compute_cross_covariance_eigenvalues(
        gram_batch_train_p: jnp.ndarray,
        gram_batch_train_q: jnp.ndarray,
        eigenvalue_regularisation: float = 1e-8,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
        use_symmetric_matrix_eigendecomposition: bool = True,
    ) -> jnp.ndarray:
        """
        Compute the eigenvalues of the covariance matrix of shape (m, m).
        Regularisation is applied to the covariance matrix before computing.
            - n is the number of training points
            - m is the number of batch points
            - d is the number of dimensions

        Args:
            gram_batch_train_p: the gram matrix of the first Gaussian measure with the batch points and training points
                                of shape (m, n)
            gram_batch_train_q: the gram matrix of the second Gaussian measure with the batch points and training points
                                of shape (m, n)
            eigenvalue_regularisation: the regularisation to add to the covariance matrix during eigenvalue computation
            is_eigenvalue_regularisation_absolute_scale: whether the regularisation is an absolute or relative scale
            use_symmetric_matrix_eigendecomposition: ensure symmetric matrices for eignedecomposition

        Returns: the eigenvalues of the covariance matrix, a vector of shape (m, 1)

        """
        if use_symmetric_matrix_eigendecomposition:
            return compute_product_eigenvalues(
                matrix_a=add_diagonal_regulariser(
                    matrix=gram_batch_train_q,
                    diagonal_regularisation=eigenvalue_regularisation,
                    is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
                ),
                matrix_b=add_diagonal_regulariser(
                    matrix=gram_batch_train_p,
                    diagonal_regularisation=eigenvalue_regularisation,
                    is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
                ),
            )
        else:
            warnings.warn(
                "covariance matrices are non-symmetric and cannot utilise GPU resources"
            )
            covariance_p_q_regularised = add_diagonal_regulariser(
                matrix=gram_batch_train_p @ gram_batch_train_q.T,
                diagonal_regularisation=eigenvalue_regularisation,
                is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
            )
            return compute_covariance_eigenvalues(covariance_p_q_regularised)

    @staticmethod
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_gaussian_wasserstein_metric(
        mean_train_p: jnp.ndarray,
        covariance_train_p_diagonal: jnp.ndarray,
        mean_train_q: jnp.ndarray,
        covariance_train_q_diagonal: jnp.ndarray,
        gram_batch_train_p: Optional[jnp.ndarray],
        gram_batch_train_q: Optional[jnp.ndarray],
        eigenvalue_regularisation: float = 1e-8,
        is_eigenvalue_regularisation_absolute_scale: bool = False,
        use_symmetric_matrix_eigendecomposition: bool = True,
        include_eigendecomposition: bool = True,
    ) -> float:
        """
        Compute the empirical Gaussian Wasserstein metric between two Gaussian measures using
        precomputed mean, covariance, and gram matrices.
                - n is the number of training points
                - m is the number of batch points
                - d is the number of dimensions
        Args:
            mean_train_p: the mean of the first Gaussian measure of shape (n, 1)
            covariance_train_p_diagonal: the covariance diagonal of the first Gaussian measure of shape (n, n)
            mean_train_q: the mean of the second Gaussian measure of shape (n, 1)
            covariance_train_q_diagonal: the covariance diagonal of the second Gaussian measure of shape (n, n)
            gram_batch_train_p: the gram matrix of the first Gaussian measure with the batch points and training points
                                of shape (m, n)
            gram_batch_train_q: the gram matrix of the second Gaussian measure with the batch points and training points
                                of shape (m, n)
            eigenvalue_regularisation: the regularisation to add to the covariance matrix during eigenvalue computation
            is_eigenvalue_regularisation_absolute_scale: whether the regularisation is an absolute or relative scale
            use_symmetric_matrix_eigendecomposition: ensure symmetric matrices for eignedecomposition
            include_eigendecomposition: whether to include the eigendecomposition term of the Gaussian wasserstein metric

        Returns: the empirical Gaussian Wasserstein metric

        """
        gaussian_wasserstein_metric = (
            jnp.mean(jnp.square(mean_train_p - mean_train_q))
            + jnp.mean(covariance_train_p_diagonal)
            + jnp.mean(covariance_train_q_diagonal)
        )
        if include_eigendecomposition:
            batch_size, train_size = gram_batch_train_p.shape
            cross_covariance_eigenvalues = GaussianWassersteinRegularisation._compute_cross_covariance_eigenvalues(
                gram_batch_train_p,
                gram_batch_train_q,
                eigenvalue_regularisation=eigenvalue_regularisation,
                is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
                use_symmetric_matrix_eigendecomposition=use_symmetric_matrix_eigendecomposition,
            )
            gaussian_wasserstein_metric -= (
                2 / jnp.sqrt(batch_size * train_size)
            ) * jnp.sum(jnp.sqrt(cross_covariance_eigenvalues))
        return jnp.float64(gaussian_wasserstein_metric)

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        gaussian_p = self.regulariser.calculate_prediction_gaussian(
            parameters=self.regulariser_parameters,
            x=x,
            full_covariance=False,
        )
        gaussian_q = self.gp.calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=False,
        )
        mean_train_p = jnp.atleast_2d(gaussian_p.mean).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        covariance_train_p_diagonal = jnp.atleast_2d(gaussian_p.covariance).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        mean_train_q = jnp.atleast_2d(gaussian_q.mean).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        covariance_train_q_diagonal = jnp.atleast_2d(gaussian_q.covariance).reshape(
            self.gp.mean.number_output_dimensions, -1
        )
        if self.include_eigendecomposition:
            gram_batch_train_p = (
                self.regulariser.calculate_prediction_gaussian_covariance(
                    parameters=self.regulariser_parameters,
                    x=x,
                    full_covariance=True,
                )
            )
            gram_batch_train_q = self.gp.calculate_prediction_gaussian_covariance(
                parameters=parameters,
                x=x,
                full_covariance=True,
            )

            gram_batch_train_p = jnp.atleast_3d(gram_batch_train_p).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
            )
            gram_batch_train_q = jnp.atleast_3d(gram_batch_train_q).reshape(
                self.gp.mean.number_output_dimensions, x.shape[0], x.shape[0]
            )
            return jnp.mean(
                jax.vmap(
                    lambda m_p, c_p, m_q, c_q, c_bt_p, c_bt_q: GaussianWassersteinRegularisation.calculate_gaussian_wasserstein_metric(
                        mean_train_p=m_p,
                        covariance_train_p_diagonal=c_p,
                        mean_train_q=m_q,
                        covariance_train_q_diagonal=c_q,
                        gram_batch_train_p=c_bt_p,
                        gram_batch_train_q=c_bt_q,
                        eigenvalue_regularisation=self.eigenvalue_regularisation,
                        is_eigenvalue_regularisation_absolute_scale=self.is_eigenvalue_regularisation_absolute_scale,
                        use_symmetric_matrix_eigendecomposition=self.use_symmetric_matrix_eigendecomposition,
                        include_eigendecomposition=self.include_eigendecomposition,
                    )
                )(
                    mean_train_p,
                    covariance_train_p_diagonal,
                    mean_train_q,
                    covariance_train_q_diagonal,
                    gram_batch_train_p,
                    gram_batch_train_q,
                )
            ).astype(jnp.float64)
        else:
            return jnp.mean(
                jax.vmap(
                    lambda m_p, c_p, m_q, c_q: GaussianWassersteinRegularisation.calculate_gaussian_wasserstein_metric(
                        mean_train_p=m_p,
                        covariance_train_p_diagonal=c_p,
                        mean_train_q=m_q,
                        covariance_train_q_diagonal=c_q,
                        gram_batch_train_p=None,
                        gram_batch_train_q=None,
                        eigenvalue_regularisation=self.eigenvalue_regularisation,
                        is_eigenvalue_regularisation_absolute_scale=self.is_eigenvalue_regularisation_absolute_scale,
                        use_symmetric_matrix_eigendecomposition=self.use_symmetric_matrix_eigendecomposition,
                        include_eigendecomposition=self.include_eigendecomposition,
                    )
                )(
                    mean_train_p,
                    covariance_train_p_diagonal,
                    mean_train_q,
                    covariance_train_q_diagonal,
                )
            ).astype(jnp.float64)
