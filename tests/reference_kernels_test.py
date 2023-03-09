from typing import Dict

import jax.numpy as jnp
import pydantic
import pytest
from jax.config import config

from src.kernels.reference_kernels import ARDKernel, StandardKernel

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "kernel,parameters,x,y,k",
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
            6.0,
            6.0,
            jnp.exp(0),
        ],
        [
            ARDKernel(number_of_dimensions=1),
            {
                "log_scaling": jnp.log(1),
                "log_lengthscales": jnp.array(-0.5),
            },
            6.0,
            6.0,
            jnp.exp(0),
        ],
    ],
)
def test_standard_kernels(
    kernel: StandardKernel,
    parameters: Dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    k: float,
):
    assert (
        kernel.calculate_kernel(kernel.generate_parameters(parameters), x=x, y=y) == k
    )


@pytest.mark.parametrize(
    "kernel,parameters,x,y,k",
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
            jnp.array([[2.7182818284590455, 1.8095777100611745]]),
        ],
    ],
)
def test_standard_kernel_grams(
    kernel: StandardKernel,
    parameters: Dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    k: float,
):
    assert jnp.array_equal(
        kernel.calculate_gram(kernel.generate_parameters(parameters), x=x, y=y), k
    )


@pytest.mark.parametrize(
    "kernel,parameters,x,y",
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
def test_missing_kernel_parameter(
    kernel: StandardKernel,
    parameters: Dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    with pytest.raises(pydantic.ValidationError):
        kernel.calculate_kernel(kernel.generate_parameters(parameters), x=x, y=y)
