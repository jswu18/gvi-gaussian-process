import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernels import (
    ReferenceKernelMock,
    ReferenceKernelParametersMock,
    calculate_reference_gram_eye_mock,
)
from src.kernels.approximate_kernels import StochasticVariationalGaussianProcessKernel
from src.parameters.kernels.approximate_kernels import (
    StochasticVariationalGaussianProcessKernelParameters,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "x_train,x_inducing,target_el_matrix_lower_triangle",
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
                    [0, 0],
                    [0, 0],
                ]
            ),
        ],
    ],
)
def test_svgp_sigma_lower_triangle_matrix(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    target_el_matrix_lower_triangle: jnp.ndarray,
):
    approximate_kernel = StochasticVariationalGaussianProcessKernel(
        reference_kernel_parameters=ReferenceKernelParametersMock(),
        log_observation_noise=jnp.log(1),
        reference_kernel=ReferenceKernelMock(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    el_matrix_lower_triangle, _ = approximate_kernel.initialise_el_matrix_parameters()
    assert jnp.array_equal(
        el_matrix_lower_triangle,
        target_el_matrix_lower_triangle,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,target_el_matrix_log_diagonal",
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
                [-0.34657359027997275, -0.34657359027997275],
            ),
        ],
    ],
)
def test_svgp_sigma_diagonal_matrix(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    target_el_matrix_log_diagonal: jnp.ndarray,
):
    approximate_kernel = StochasticVariationalGaussianProcessKernel(
        reference_kernel_parameters=ReferenceKernelParametersMock(),
        log_observation_noise=jnp.log(1),
        reference_kernel=ReferenceKernelMock(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    _, el_matrix_log_diagonal = approximate_kernel.initialise_el_matrix_parameters()
    assert jnp.array_equal(
        el_matrix_log_diagonal,
        target_el_matrix_log_diagonal,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,el_matrix_lower_triangle,el_matrix_log_diagonal,gram",
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
                    [0, 0],
                    [0.2, 0],
                ]
            ),
            jnp.array(
                [4.3, 2.2],
            ),
            5516.760470427696 * jnp.ones((3, 3)),
        ],
    ],
)
def test_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    el_matrix_lower_triangle: jnp.ndarray,
    el_matrix_log_diagonal: jnp.ndarray,
    gram: float,
):
    approximate_kernel = StochasticVariationalGaussianProcessKernel(
        reference_kernel_parameters=ReferenceKernelParametersMock(),
        log_observation_noise=jnp.log(1),
        reference_kernel=ReferenceKernelMock(),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    assert jnp.array_equal(
        approximate_kernel.calculate_gram(
            StochasticVariationalGaussianProcessKernelParameters(
                el_matrix_lower_triangle=el_matrix_lower_triangle,
                el_matrix_log_diagonal=el_matrix_log_diagonal,
            ),
            x=x_train,
        ),
        gram,
    )
