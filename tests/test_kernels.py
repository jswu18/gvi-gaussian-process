from typing import Dict

import jax.numpy as jnp
import pydantic
import pytest
from jax.config import config
from mock import Mock

from mockers.kernel import (
    MockKernel,
    MockKernelParameters,
    calculate_reference_gram_eye_mock,
)
from src.kernels import CustomKernel, MultiOutputKernel
from src.kernels.approximate import DecomposedSVGPKernel
from src.kernels.standard import ARDKernel

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            ARDKernel(number_of_dimensions=3),
            {
                "log_scaling": 0.5,
                "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
            },
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.5, 2.5, 3.5]),
            1.8095777100611745,
        ],
        [
            ARDKernel(number_of_dimensions=1),
            {
                "log_scaling": jnp.log(1),
                "log_lengthscales": jnp.array(-0.5),
            },
            jnp.array(6.0),
            jnp.array(6.0),
            jnp.exp(0),
        ],
        [
            ARDKernel(number_of_dimensions=1),
            {
                "log_scaling": jnp.log(1),
                "log_lengthscales": jnp.array(-0.5),
            },
            jnp.array(6.0),
            jnp.array(6.0),
            jnp.exp(0),
        ],
    ],
)
def test_ard_kernels(
    kernel: ARDKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    assert (
        kernel.calculate_kernel(kernel.generate_parameters(parameters), x1=x1, x2=x2)
        == k
    )


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            ARDKernel(number_of_dimensions=3),
            {
                "log_scaling": 0.5,
                "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
            },
            jnp.array([[1.0, 2.0, 3.0]]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((1, 2)),
        ],
    ],
)
def test_ard_kernel_grams(
    kernel: ARDKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    kernel.calculate_kernel = Mock(return_value=1)
    assert jnp.array_equal(
        kernel.calculate_gram(kernel.generate_parameters(parameters), x1=x1, x2=x2), k
    )


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            ARDKernel(number_of_dimensions=3),
            {
                "log_scaling": 0.5,
                "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
            },
            jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((2,)),
        ],
    ],
)
def test_ard_kernel_gram_diagonal(
    kernel: ARDKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    kernel.calculate_kernel = Mock(return_value=1)
    assert jnp.array_equal(
        kernel.calculate_gram(
            kernel.generate_parameters(parameters), x1=x1, x2=x2, full_covariance=False
        ),
        k,
    )


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2",
    [
        [
            ARDKernel(number_of_dimensions=1),
            {
                "log_scaling": jnp.log(1),
            },
            jnp.array(6.0),
            jnp.array(6.0),
        ],
    ],
)
def test_missing_ard_kernel_parameter(
    kernel: ARDKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
):
    with pytest.raises(pydantic.ValidationError):
        kernel.calculate_kernel(kernel.generate_parameters(parameters), x1=x1, x2=x2)


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            CustomKernel(
                kernel_function=lambda parameters, x, y: jnp.ones(
                    (x.shape[0], y.shape[0])
                )
            ),
            None,
            jnp.array([[1.0, 2.0, 3.0]]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((1, 2)),
        ],
        [
            CustomKernel(
                kernel_function=lambda parameters, x, y: jnp.ones(
                    (x.shape[0], y.shape[0])
                ),
                preprocess_function=lambda x: x.reshape(x.shape[0], -1),
            ),
            None,
            jnp.array([[[1.0], [2.0], [3.0]]]),
            jnp.array(
                [
                    [[1.0], [2.0], [3.0]],
                    [[1.5], [2.5], [3.5]],
                ]
            ),
            jnp.ones((1, 2)),
        ],
    ],
)
def test_custom_kernel_grams(
    kernel: CustomKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    assert jnp.array_equal(
        kernel.calculate_gram(
            kernel.Parameters.construct(custom=parameters), x1=x1, x2=x2
        ),
        k,
    )


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            MultiOutputKernel(
                kernels=[
                    MockKernel(),
                    MockKernel(),
                    MockKernel(),
                ]
            ),
            {
                "kernels": [
                    MockKernelParameters(),
                    MockKernelParameters(),
                    MockKernelParameters(),
                ]
            },
            jnp.array([[1.0, 2.0, 3.0]]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((3, 1, 2)),
        ],
    ],
)
def test_multi_output_kernel_grams(
    kernel: CustomKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    if parameters is None:
        parameters = {}
    assert jnp.array_equal(
        kernel.calculate_gram(kernel.generate_parameters(parameters), x1=x1, x2=x2), k
    )


@pytest.mark.parametrize(
    "kernel,parameters,x,k",
    [
        [
            MultiOutputKernel(
                kernels=[
                    MockKernel(),
                    MockKernel(),
                    MockKernel(),
                ]
            ),
            {
                "kernels": [
                    MockKernelParameters(),
                    MockKernelParameters(),
                    MockKernelParameters(),
                ]
            },
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((3, 2)),
        ],
    ],
)
def test_multi_output_kernel_diagonal_grams(
    kernel: CustomKernel,
    parameters: Dict,
    x: jnp.ndarray,
    k: float,
):
    if parameters is None:
        parameters = {}
    assert jnp.array_equal(
        kernel.calculate_gram(
            kernel.generate_parameters(parameters),
            x1=x,
            x2=x,
            full_covariance=False,
        ),
        k,
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
    svgp_kernel = DecomposedSVGPKernel(
        reference_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        reference_kernel=MockKernel(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    el_matrix_lower_triangle, _ = svgp_kernel.initialise_el_matrix_parameters()
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
def test_svgp_sigma_log_diagonal(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    target_el_matrix_log_diagonal: jnp.ndarray,
):
    svgp_kernel = DecomposedSVGPKernel(
        reference_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        reference_kernel=MockKernel(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    _, el_matrix_log_diagonal = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.array_equal(
        el_matrix_log_diagonal,
        target_el_matrix_log_diagonal,
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
def test_svgp_kernel_grams(
    el_matrix_lower_triangle: jnp.ndarray,
    el_matrix_log_diagonal: jnp.ndarray,
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    k: jnp.ndarray,
):
    svgp_kernel = DecomposedSVGPKernel(
        reference_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        reference_kernel=MockKernel(calculate_reference_gram_eye_mock),
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
    "x_train,x_inducing,x,actual_el_matrix_lower_triangle",
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
                [[0.0, 0.0], [0.0, 0.0]],
            ),
        ],
    ],
)
def test_initialise_el_matrix_lower_triangle_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    actual_el_matrix_lower_triangle: jnp.ndarray,
):
    svgp_kernel = DecomposedSVGPKernel(
        reference_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        reference_kernel=MockKernel(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    el_matrix_lower_triangle, _ = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.allclose(
        el_matrix_lower_triangle,
        actual_el_matrix_lower_triangle,
    )


@pytest.mark.parametrize(
    "x_train,x_inducing,x,actual_el_matrix_log_diagonal",
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
            jnp.array([-0.34657359, -0.34657359]),
        ],
    ],
)
def test_initialise_el_matrix_lower_triangle_svgp_kernel_grams(
    x_train: jnp.ndarray,
    x_inducing: jnp.ndarray,
    x: jnp.ndarray,
    actual_el_matrix_log_diagonal: jnp.ndarray,
):
    svgp_kernel = DecomposedSVGPKernel(
        reference_kernel_parameters=MockKernelParameters(),
        log_observation_noise=jnp.log(1),
        reference_kernel=MockKernel(calculate_reference_gram_eye_mock),
        inducing_points=x_inducing,
        training_points=x_train,
    )
    _, el_matrix_log_diagonal = svgp_kernel.initialise_el_matrix_parameters()
    assert jnp.allclose(
        el_matrix_log_diagonal,
        actual_el_matrix_log_diagonal,
    )
