import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.gaussian_measures import GaussianMeasure
from src.utils import add_diagonal_regulariser, compute_covariance_eigenvalues


def _compute_cross_covariance_eigenvalues(
    p: GaussianMeasure,
    q: GaussianMeasure,
    p_parameters: FrozenDict,
    q_parameters: FrozenDict,
    x_batch: jnp.ndarray,
    x_train: jnp.ndarray,
    eigenvalue_regularisation: float = 0.0,
    is_eigenvalue_regularisation_absolute_scale: bool = False,
) -> jnp.ndarray:
    gram_batch_train_p = p.covariance(x=x_batch, y=x_train, parameters=p_parameters)
    gram_batch_train_q = q.covariance(x=x_batch, y=x_train, parameters=q_parameters)
    covariance_p_q = gram_batch_train_p @ gram_batch_train_q.T
    covariance_p_q_regularised = add_diagonal_regulariser(
        matrix=covariance_p_q,
        diagonal_regularisation=eigenvalue_regularisation,
        is_diagonal_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
    )
    return compute_covariance_eigenvalues(covariance_p_q_regularised)


def gaussian_wasserstein_metric(
    p: GaussianMeasure,
    q: GaussianMeasure,
    p_parameters: FrozenDict,
    q_parameters: FrozenDict,
    x_batch: jnp.ndarray,
    x_train: jnp.ndarray,
    eigenvalue_regularisation: float = 0.0,
    is_eigenvalue_regularisation_absolute_scale: bool = False,
) -> float:
    train_size = x_train.shape[0]
    batch_size = x_batch.shape[0]

    mean_train_p, covariance_train_p = p.mean_and_covariance(
        x=x_train, parameters=p_parameters
    )
    mean_train_q, covariance_train_q = q.mean_and_covariance(
        x=x_train, parameters=q_parameters
    )
    cross_covariance_eigenvalues = _compute_cross_covariance_eigenvalues(
        p=p,
        q=q,
        p_parameters=p_parameters,
        q_parameters=q_parameters,
        x_batch=x_batch,
        x_train=x_train,
        eigenvalue_regularisation=eigenvalue_regularisation,
        is_eigenvalue_regularisation_absolute_scale=is_eigenvalue_regularisation_absolute_scale,
    )
    return (
        jnp.mean((mean_train_p - mean_train_q) ** 2)
        + jnp.mean(jnp.diagonal(covariance_train_p))
        + jnp.mean(jnp.diagonal(covariance_train_q))
        - (2 / jnp.sqrt(batch_size * train_size))
        * jnp.sum(jnp.sqrt(cross_covariance_eigenvalues))
    )
