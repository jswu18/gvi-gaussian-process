from typing import Dict

import jax.numpy as jnp
import pytest
from jax.config import config
from mock import Mock

from mockers.kernel import MockKernel, MockKernelParameters
from src.kernels import CustomKernel, MultiOutputKernel
from src.kernels.non_stationary import InnerProductKernel, PolynomialKernel
from src.kernels.standard import ARDKernel

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_scaling,x1,x2,k",
    [
        [
            jnp.log(2.4),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.5, 2.5, 3.5]),
            40.8,
        ],
        [
            jnp.log(2.2),
            jnp.array(6.0),
            jnp.array(6.0),
            79.2,
        ],
        [
            jnp.log(2.0),
            jnp.array(6.0),
            jnp.array(6.0),
            72,
        ],
    ],
)
def test_inner_product_kernels(
    log_scaling: float,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    kernel = InnerProductKernel()
    parameters = InnerProductKernel.Parameters(log_scaling=log_scaling)
    assert jnp.isclose(kernel.calculate_kernel(parameters, x1=x1, x2=x2), k)


@pytest.mark.parametrize(
    "kernel,parameters,x1,x2,k",
    [
        [
            PolynomialKernel(polynomial_degree=3),
            {
                "log_constant": 0.5,
                "log_scaling": 3.2,
            },
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.5, 2.5, 3.5]),
            73403079.4946829,
        ],
        [
            PolynomialKernel(polynomial_degree=1),
            {
                "log_constant": 0.5,
                "log_scaling": 3.2,
            },
            jnp.array(6.0),
            jnp.array(6.0),
            884.81980837,
        ],
        [
            PolynomialKernel(polynomial_degree=1.5),
            {
                "log_constant": 0.5,
                "log_scaling": 3.2,
            },
            jnp.array(6.0),
            jnp.array(6.0),
            26319.78000332,
        ],
    ],
)
def test_polynomial_kernels(
    kernel: PolynomialKernel,
    parameters: Dict,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    k: float,
):
    assert jnp.isclose(
        kernel.calculate_kernel(kernel.generate_parameters(parameters), x1=x1, x2=x2), k
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
    inner_product_kernel = InnerProductKernel()
    with pytest.raises(AssertionError):
        kernel.calculate_kernel(
            inner_product_kernel.generate_parameters(parameters), x1=x1, x2=x2
        )


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
