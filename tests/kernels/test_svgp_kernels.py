import pytest
from jax import numpy as jnp
from jax.config import config

from mockers.kernel import (
    MockKernel,
    MockKernelParameters,
    calculate_regulariser_gram_eye_mock,
)
from src.kernels.approximate import (
    CholeskySVGPKernel,
    DiagonalSVGPKernel,
    KernelisedSVGPKernel,
    LogSVGPKernel,
)

config.update("jax_enable_x64", True)


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
def test_svgp_sigma_log_diagonal(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    target_el_matrix_log_diagonal: jnp.ndarray,
):
    svgp_kernel = CholeskySVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    _, el_matrix_log_diagonal = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.array_equal(
        el_matrix_log_diagonal,
        target_el_matrix_log_diagonal,
    )


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
    svgp_kernel = CholeskySVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    el_matrix_lower_triangle, _ = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.array_equal(
        el_matrix_lower_triangle,
        target_el_matrix_lower_triangle,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,x,actual_log_diagonal",
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
            jnp.array([-0.69314724, -0.69314724]),
        ],
    ],
)
def test_initialise_diagonal_svgp_kernel(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    actual_log_diagonal: jnp.ndarray,
):
    svgp_kernel = DiagonalSVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    log_diagonal = svgp_kernel.initialise_diagonal_parameters()
    assert jnp.allclose(
        log_diagonal,
        actual_log_diagonal,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,x,actual_el_matrix",
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
            jnp.array(
                [[-0.34657362, -11.512925], [-11.512925, -0.34657362]],
            ),
        ],
    ],
)
def test_initialise_el_matrix_log_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    actual_el_matrix: jnp.ndarray,
):
    svgp_kernel = LogSVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    el_matrix = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.allclose(
        el_matrix,
        actual_el_matrix,
    )


@pytest.mark.parametrize(
    "el_matrix_lower_triangle,el_matrix_log_diagonal,x_train,x_inducing,x,k",
    [
        [
            jnp.array(
                [[0.0, 0.0], [0.0, 0.0]],
            ),
            jnp.array([-0.34657359, -0.34657359]),
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
            jnp.array(
                [
                    [0.5000099999000007, 0],
                    [0, 0.5000099999000007],
                ]
            ),
        ],
    ],
)
def test_cholesky_svgp_kernel_grams(
    el_matrix_lower_triangle: jnp.ndarray,
    el_matrix_log_diagonal: jnp.ndarray,
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    svgp_kernel = CholeskySVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    parameters = svgp_kernel.generate_parameters(
        {
            "el_matrix_lower_triangle": el_matrix_lower_triangle,
            "el_matrix_log_diagonal": el_matrix_log_diagonal,
        }
    )
    assert jnp.allclose(
        svgp_kernel.calculate_gram(
            parameters=parameters, x1=x, x2=x, full_covariance=True
        ),
        k,
    )


@pytest.mark.parametrize(
    "log_el_matrix,x_train,x_inducing,x,k",
    [
        [
            jnp.array(
                [[-0.34657362, -11.512925], [-11.512925, -0.34657362]],
            ),
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
            jnp.array(
                [
                    [2.5000998e-01, 1.4142140e-05],
                    [1.4142140e-05, 2.5000998e-01],
                ]
            ),
        ],
    ],
)
def test_log_el_svgp_kernel_grams(
    log_el_matrix: jnp.ndarray,
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    svgp_kernel = LogSVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    parameters = svgp_kernel.generate_parameters(
        {
            "log_el_matrix": log_el_matrix,
        }
    )
    assert jnp.allclose(
        svgp_kernel.calculate_gram(
            parameters=parameters, x1=x, x2=x, full_covariance=True
        ),
        k,
    )


@pytest.mark.parametrize(
    "log_el_matrix_diagonal,x_train,x_inducing,x,k",
    [
        [
            jnp.array([-0.69314724, -0.69314724]),
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
            jnp.array([[0.50001, 0.0], [0.0, 0.50001]]),
        ],
    ],
)
def test_diagonal_svgp_kernel_grams(
    log_el_matrix_diagonal: jnp.ndarray,
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    svgp_kernel = DiagonalSVGPKernel(
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    parameters = svgp_kernel.generate_parameters(
        {
            "log_el_matrix_diagonal": log_el_matrix_diagonal,
        }
    )
    assert jnp.allclose(
        svgp_kernel.calculate_gram(
            parameters=parameters, x1=x, x2=x, full_covariance=True
        ),
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
            jnp.array([[1.00001, 1.0], [1.0, 1.00001]]),
        ],
    ],
)
def test_kernelised_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    svgp_kernel = KernelisedSVGPKernel(
        base_kernel=MockKernel(),
        regulariser_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        regulariser_kernel=MockKernel(calculate_regulariser_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    parameters = svgp_kernel.generate_parameters(
        {
            "base_kernel": MockKernelParameters(),
        }
    )
    assert jnp.allclose(
        svgp_kernel.calculate_gram(
            parameters=parameters, x1=x, x2=x, full_covariance=True
        ),
        k,
    )
