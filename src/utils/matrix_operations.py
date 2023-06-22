import logging

import jax.numpy as jnp
import jax.scipy as jsp

# from jax.experimental import host_callback


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
     This implementation is not available for gpus.
    Follows from:
    https://github.com/google/neural-tangents/blob/5d38d3e97a2e251c37bb1ba44a89cbb8565a5459/neural_tangents/_src/predict.py#L1266
    Args:
        matrix: a covariance matrix of shape (n, n)
        logging_warning_threshold: the threshold for logging a warning of the min to max eigenvalue ratio

    Returns: the eigenvalues of the covariance matrix, a vector of shape (n, 1)

    """
    covariance_eigenvalues = jnp.linalg.eigvals(matrix)  # doesn't work for GPU
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


def compute_product_eigenvalues(
    matrix_a: jnp.ndarray,
    matrix_b: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the eigenvalues of a product of two symmetric matrices:
        eig(A*B))
    We will use the fact that:
        eig(A*B) = eig(sqrt(B)*A*sqrt(B))
    where A and B are both symmetric matrices.
    This implementation uses jnp.linalg.eigh which has GPU implementation
    Args:
        matrix_a: symmetric matrix of shape (n, n)
        matrix_b: symmetric matrix of shape (n, n)

    Returns: the eigenvalues of the product matrix_a*matrix_b

    """
    # check symmetric matrices
    assert is_symmetric(matrix_a), f"non-symmetric matrix {matrix_a=}"
    assert is_symmetric(matrix_a), f"non-symmetric matrix {matrix_b=}"

    matrix_a_eigenvalues, _ = jnp.linalg.eigh(matrix_a)
    matrix_b_sqrt = jsp.linalg.sqrtm(matrix_b)
    matrix_b_sqrt_eigenvalues, _ = jnp.linalg.eigh(matrix_b_sqrt)
    covariance_eigenvalues = jnp.prod(
        jnp.stack(
            (matrix_b_sqrt_eigenvalues, matrix_a_eigenvalues, matrix_b_sqrt_eigenvalues)
        ),
        axis=0,
    )
    return jnp.clip(
        covariance_eigenvalues,
        a_min=0,
        a_max=None,
    ).real


def is_symmetric(matrix: jnp.ndarray) -> bool:
    """
    Check if matrix is symmetric
    Args:
        matrix: array

    Returns: boolean

    """
    return bool(jnp.allclose(matrix, matrix.T))
