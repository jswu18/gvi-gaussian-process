import jax.numpy as jnp
import pytest

from src.utils.matrix_operations import (
    add_diagonal_regulariser,
    compute_covariance_eigenvalues,
    compute_product_eigenvalues,
)


@pytest.mark.parametrize(
    "matrix,eigenvalues",
    [
        [
            jnp.array(
                [
                    [1, 0.5],
                    [0.5, 1],
                ]
            ),
            jnp.array(
                [0.5, 1.5],
            ),
        ],
        [
            jnp.array(
                [
                    [15, 2.4],
                    [2.4, 15],
                ]
            ),
            jnp.array(
                [12.6, 17.4],
            ),
        ],
    ],
)
def test_compute_covariance_eigenvalues(
    matrix: jnp.ndarray,
    eigenvalues: jnp.ndarray,
):
    assert jnp.allclose(
        jnp.sort(compute_covariance_eigenvalues(matrix)),
        eigenvalues,
        rtol=1e-05,
        atol=1e-08,
    )


@pytest.mark.parametrize(
    "matrix_a,matrix_b",
    [
        [
            jnp.array(
                [
                    [1, 0.5],
                    [0.5, 1],
                ]
            ),
            jnp.array(
                [
                    [3, 1.5],
                    [1.5, 3],
                ]
            ),
        ],
        [
            jnp.array(
                [
                    [15, 2.4],
                    [2.4, 15],
                ]
            ),
            jnp.array(
                [
                    [2.3, 0],
                    [0, 2.3],
                ]
            ),
        ],
    ],
)
def test_compute_product_eigenvalues(
    matrix_a: jnp.ndarray,
    matrix_b: jnp.ndarray,
):
    assert jnp.allclose(
        jnp.sort(compute_covariance_eigenvalues(jnp.dot(matrix_a, matrix_b))),
        jnp.sort(compute_product_eigenvalues(matrix_a, matrix_b)),
        rtol=1e-05,
        atol=1e-08,
    )


@pytest.mark.parametrize(
    "matrix,diagonal_regularisation,is_diagonal_regularisation_absolute_scale,regularised_matrix",
    [
        [
            jnp.array(
                [
                    [1, 0.5],
                    [0.5, 1],
                ]
            ),
            1e-1,
            False,
            jnp.array(
                [
                    [1 + 1e-1, 0.5],
                    [0.5, 1 + 1e-1],
                ]
            ),
        ],
        [
            jnp.array(
                [
                    [0.1, 0.5],
                    [0.5, 1],
                ]
            ),
            1e-1,
            True,
            jnp.array(
                [
                    [0.2, 0.5],
                    [0.5, 1.1],
                ]
            ),
        ],
    ],
)
def test_add_diagonal_regulariser(
    matrix: jnp.ndarray,
    diagonal_regularisation: float,
    is_diagonal_regularisation_absolute_scale: bool,
    regularised_matrix: jnp.ndarray,
):
    assert jnp.allclose(
        add_diagonal_regulariser(
            matrix=matrix,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        ),
        regularised_matrix,
    )
