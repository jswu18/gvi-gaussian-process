from typing import Dict

import jax.numpy as jnp
import pydantic
import pytest
from jax.config import config
from mock import Mock

from src.kernels import (
    NeuralNetworkGaussianProcessKernel,
    NeuralNetworkGaussianProcessKernelParameters,
)
from src.kernels.approximate import (
    StochasticVariationalKernel,
    StochasticVariationalKernelParameters,
)
from src.kernels.standard import ARDKernel, ARDKernelParameters

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
