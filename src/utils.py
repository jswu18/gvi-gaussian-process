import logging

import jax.numpy as jnp


def add_diagonal_regulariser(
    matrix: jnp.ndarray,
    diagonal_regularisation: float,
    is_diagonal_regularisation_absolute_scale: bool,
) -> jnp.ndarray:
    # https://github.com/google/neural-tangents/blob/5d38d3e97a2e251c37bb1ba44a89cbb8565a5459/neural_tangents/_src/predict.py#L1216
    dimension = matrix.shape[0]
    if not is_diagonal_regularisation_absolute_scale:
        diagonal_regularisation *= jnp.trace(matrix) / dimension
    return matrix + diagonal_regularisation * jnp.eye(dimension)


def compute_covariance_eigenvalues(
    matrix: jnp.ndarray, logging_warning_threshold: float = 5e-2
) -> jnp.ndarray:
    covariance_eigenvalues, _ = jnp.linalg.eigh(matrix)
    # minimum_covariance_eigenvalue = jnp.min(covariance_eigenvalues)
    # if minimum_covariance_eigenvalue < 0:
    #     spectrum_ratio = jnp.abs(minimum_covariance_eigenvalue) / jnp.max(
    #         covariance_eigenvalues
    #     )
    #     if spectrum_ratio > logging_warning_threshold:
    #         logging.warning(
    #             f"Covariance has negatives. {spectrum_ratio=}, {logging_warning_threshold=}, {covariance_eigenvalues=}"
    #         )

    # following
    # https://github.com/google/neural-tangents/blob/5d38d3e97a2e251c37bb1ba44a89cbb8565a5459/neural_tangents/_src/predict.py#L1266
    return jnp.clip(
        covariance_eigenvalues,
        a_min=0,
        a_max=None,
    )
