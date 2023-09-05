import pytest
from jax import numpy as jnp
from jax.config import config

from mockers.kernel import (
    MockKernel,
    MockKernelParameters,
    calculate_regulariser_gram_eye_mock,
)
from src.kernels.approximate import FixedSparsePosteriorKernel, SparsePosteriorKernel

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "x_train,x_inducing,x,k",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [5.0, 1.0, 9.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [5.0, 1.0, 9.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [5.3, 5.0, 6.0],
                    [2.5, 4.5, 2.5],
                ]
            ),
            jnp.array([[9.9999e-06, 0.0000e00], [0.0000e00, 9.9999e-06]]),
        ],
    ],
)
def test_sparse_posterior_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    kernel = SparsePosteriorKernel(
        base_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        diagonal_regularisation=1e-5,
        is_diagonal_regularisation_absolute_scale=False,
    )
    parameters = kernel.generate_parameters(
        {
            "base_kernel": MockKernelParameters(),
        }
    )
    assert jnp.allclose(
        kernel.calculate_gram(parameters=parameters, x1=x, x2=x, full_covariance=True),
        k,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,x,k",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [5.0, 1.0, 9.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [5.0, 1.0, 9.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [5.3, 5.0, 6.0],
                    [2.5, 4.5, 2.5],
                ]
            ),
            jnp.array([[9.9999e-06, 0.0000e00], [0.0000e00, 9.9999e-06]]),
        ],
    ],
)
def test_fixed_sparse_posterior_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    kernel = FixedSparsePosteriorKernel(
        base_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        regulariser_kernel_parameters=MockKernelParameters(),
        inducing_points=x_inducing,
        diagonal_regularisation=1e-5,
        is_diagonal_regularisation_absolute_scale=False,
    )
    parameters = kernel.generate_parameters(
        {
            "base_kernel": MockKernelParameters(),
        }
    )
    assert jnp.allclose(
        kernel.calculate_gram(parameters=parameters, x1=x, x2=x, full_covariance=True),
        k,
    )
