import logging

import jax.numpy as jnp
from jax.experimental import host_callback


def add_diagonal_regulariser(
    matrix: jnp.ndarray,
    diagonal_regularisation: float,
    is_diagonal_regularisation_absolute_scale: bool,
) -> jnp.ndarray:
    """
    Add a regularisation to the diagonal of a matrix.
    Follows from:
    https://github.com/google/neural-tangents/blob/5d38d3e97a2e251c37bb1ba44a89cbb8565a5459/neural_tangents/_src/predict.py#L1216

    Args:
        matrix: a matrix of shape (n, n)
        diagonal_regularisation: the regularisation to add to the diagonal
        is_diagonal_regularisation_absolute_scale: whether the regularisation is an absolute scale or a relative scale

    Returns: the regularised matrix of shape (n, n)

    """
    dimension = matrix.shape[0]
    if not is_diagonal_regularisation_absolute_scale:
        diagonal_regularisation *= jnp.trace(matrix) / dimension
    return matrix + diagonal_regularisation * jnp.eye(dimension)


def _eigenvalue_warning(values_to_check, _) -> None:
    (
        minimum_covariance_eigenvalue,
        spectrum_ratio,
        logging_warning_threshold,
        covariance_eigenvalues,
    ) = values_to_check
    if minimum_covariance_eigenvalue < 0 and spectrum_ratio > logging_warning_threshold:
        logging.warning(
            f"Covariance has negatives. "
            f"{spectrum_ratio=}, {logging_warning_threshold=}, "
            f"{jnp.min(covariance_eigenvalues)=}, {jnp.max(covariance_eigenvalues)=}"
        )


def compute_covariance_eigenvalues(
    matrix: jnp.ndarray, logging_warning_threshold: float = 5e-2
) -> jnp.ndarray:
    """
    Compute the eigenvalues of a covariance matrix. Because the matrix is a covariance matrix, we expect
     the eigenvalues to be positive. Thus, for numerical stability the eigenvalues are clipped to be non-negative.
    Follows from:
    https://github.com/google/neural-tangents/blob/5d38d3e97a2e251c37bb1ba44a89cbb8565a5459/neural_tangents/_src/predict.py#L1266
    Args:
        matrix: a covariance matrix of shape (n, n)
        logging_warning_threshold: the threshold for logging a warning of the min to max eigenvalue ratio

    Returns: the eigenvalues of the covariance matrix, a vector of shape (n, 1)

    """
    covariance_eigenvalues = jnp.linalg.eigvals(matrix)  # doesn't work for GPU
    # covariance_eigenvalues, _ = jnp.linalg.eigh(matrix) # works for GPU but requires symmetric matrix
    minimum_covariance_eigenvalue = jnp.min(covariance_eigenvalues)
    spectrum_ratio = jnp.abs(minimum_covariance_eigenvalue) / jnp.max(
        covariance_eigenvalues
    )
    # host_callback.id_tap(
    #     _eigenvalue_warning,
    #     (minimum_covariance_eigenvalue, spectrum_ratio, logging_warning_threshold, covariance_eigenvalues),
    # )
    return jnp.clip(
        covariance_eigenvalues,
        a_min=0,
        a_max=None,
    ).real
