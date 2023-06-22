import warnings
from typing import Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.gaussian_measures.gaussian_measures import GaussianMeasure
from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.utils.matrix_operations import (
    add_diagonal_regulariser,
    compute_covariance_eigenvalues,
    compute_product_eigenvalues,
)


def _compute_cross_covariance_eigenvalues(
    p: GaussianMeasure,
    q: GaussianMeasure,
    p_parameters: GaussianMeasureParameters,
    q_parameters: GaussianMeasureParameters,
    x_batch: jnp.ndarray,
    x_train: jnp.ndarray,
    eigenvalue_regularisation: float = 0.0,
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
        p: the first Gaussian measure
        q: the second Gaussian measure
        p_parameters: the parameters of the first Gaussian measure
        q_parameters: the parameters of the second Gaussian measure
        x_batch: a batch of data points of shape (m, d)
        x_train: the training data points of shape (n, d)
        eigenvalue_regularisation: the regularisation to add to the covariance matrix during eigenvalue computation
        is_eigenvalue_regularisation_absolute_scale: whether the regularisation is an absolute or relative scale
        use_symmetric_matrix_eigendecomposition: ensure symmetric matrices for eignedecomposition

    Returns: the eigenvalues of the covariance matrix, a vector of shape (m, 1)

    """
    gram_batch_train_p = p.calculate_covariance(
        x=x_batch, y=x_train, parameters=p_parameters
    )
    gram_batch_train_q = q.calculate_covariance(
        x=x_batch, y=x_train, parameters=q_parameters
    )
    if use_symmetric_matrix_eigendecomposition:
        return compute_product_eigenvalues(gram_batch_train_q, gram_batch_train_p)
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


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def compute_gaussian_wasserstein_metric(
    p: GaussianMeasure,
    q: GaussianMeasure,
    p_parameters: Union[FrozenDict, Dict, GaussianMeasureParameters],
    q_parameters: Union[FrozenDict, Dict, GaussianMeasureParameters],
    x_batch: jnp.ndarray,
    x_train: jnp.ndarray,
    eigenvalue_regularisation: float = 0.0,
    is_eigenvalue_regularisation_absolute_scale: bool = False,
    use_symmetric_matrix_eigendecomposition: bool = True,
) -> float:
    """
    Compute the empirical Gaussian Wasserstein metric between two Gaussian measures.
            - n is the number of training points
            - m is the number of batch points
            - d is the number of dimensions

    Args:
        p: the first Gaussian measure
        q: the second Gaussian measure
        p_parameters: a dictionary or Pydantic model containing the parameters of the first Gaussian measure
                        a dictionary is required for jit compilation which is converted if necessary
        q_parameters: a dictionary or Pydantic model containing the parameters of the second Gaussian measure
                        a dictionary is required for jit compilation which is converted if necessary
        x_batch: a batch of data points of shape (m, d)
        x_train: the training data points of shape (n, d)
        eigenvalue_regularisation: the regularisation to add to the covariance matrix during eigenvalue computation
        is_eigenvalue_regularisation_absolute_scale: whether the regularisation is an absolute or relative scale
        use_symmetric_matrix_eigendecomposition: ensure symmetric matrices for eignedecomposition

    Returns: the Gaussian Wasserstein metric between the two Gaussian measures, a scalar

    """
    train_size = x_train.shape[0]
    batch_size = x_batch.shape[0]

    # convert to Pydantic models if necessary
    if not isinstance(p_parameters, p.Parameters):
        p_parameters = p.generate_parameters(p_parameters)
    if not isinstance(q_parameters, q.Parameters):
        q_parameters = q.generate_parameters(q_parameters)

    mean_train_p = p.calculate_mean(x=x_train, parameters=p_parameters)
    covariance_train_p = p.calculate_covariance(x=x_train, parameters=p_parameters)

    mean_train_q = q.calculate_mean(x=x_train, parameters=q_parameters)
    covariance_train_q = q.calculate_covariance(x=x_train, parameters=q_parameters)

    cross_covariance_eigenvalues = _compute_cross_covariance_eigenvalues(
        p=p,
        q=q,
        p_parameters=p_parameters,
        q_parameters=q_parameters,
        x_batch=x_batch,
        x_train=x_train,
        eigenvalue_regularisation=eigenvalue_regularisation,
        is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
        use_symmetric_matrix_eigendecomposition=use_symmetric_matrix_eigendecomposition,
    )
    return (
        jnp.mean((mean_train_p - mean_train_q) ** 2)
        + jnp.mean(jnp.diagonal(covariance_train_p))
        + jnp.mean(jnp.diagonal(covariance_train_q))
        - (2 / jnp.sqrt(batch_size * train_size))
        * jnp.sum(jnp.sqrt(cross_covariance_eigenvalues))
    )
