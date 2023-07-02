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
    "x_train,x_inducing,log_el_matrix",
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
                    [-0.34657359027997275, -23.025850929940457],
                    [-23.025850929940457, -0.34657359027997275],
                ]
            ),
        ],
    ],
)
def test_svgp_sigma_matrix(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    log_el_matrix: jnp.ndarray,
):
    approximate_kernel = StochasticVariationalGaussianProcessKernel(
        reference_kernel_parameters=ReferenceKernelParametersMock(),
        log_observation_noise=jnp.log(1),
        reference_kernel=ReferenceKernelMock(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    assert jnp.array_equal(
        approximate_kernel.initialise_log_el_matrix(),
        log_el_matrix,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,log_el_matrix,gram",
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
                    [5.0, 0.1],
                    [0.2, 2.5],
                ]
            ),
            22535.395464414618 * jnp.ones((3, 3)),
        ],
    ],
)
def test_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    log_el_matrix: jnp.ndarray,
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
                log_el_matrix=log_el_matrix,
            ),
            x=x_train,
        ),
        gram,
    )
