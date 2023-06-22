import jax.numpy as jnp
import pytest

from src.utils.matrix_operations import (
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
                [0.49999994, 1.5],
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
    assert jnp.array_equal(
        jnp.sort(compute_covariance_eigenvalues(matrix)),
        eigenvalues,
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
