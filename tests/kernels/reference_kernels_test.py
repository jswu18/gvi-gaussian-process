import jax.numpy as jnp
import numpy as np
import pytest
from flax.core.frozen_dict import FrozenDict
from jax.config import config

from src.kernels.reference_kernels import ARDKernel, StandardKernel

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "kernel,parameters,x,y,k",
    [
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": 0.5,
                    "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
                }
            ),
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([1.5, 2.5, 3.5]),
            1.8095777100611745,
        ],
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": jnp.log(1),
                    "log_lengthscales": -0.5,
                }
            ),
            6.0,
            6.0,
            jnp.exp(0),
        ],
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": jnp.log(1),
                    "log_lengthscales": -0.5,
                }
            ),
            6.0,
            None,
            jnp.exp(0),
        ],
    ],
)
def test_standard_kernels(
    kernel: StandardKernel,
    parameters: FrozenDict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    k: float,
):
    assert kernel.calculate_kernel(parameters, x=x, y=y) == k


@pytest.mark.parametrize(
    "kernel,parameters,x,y,k",
    [
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": 0.5,
                    "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
                }
            ),
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
    parameters: FrozenDict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    k: float,
):
    assert jnp.array_equal(kernel.calculate_gram(parameters=parameters, x=x, y=y), k)


@pytest.mark.parametrize(
    "kernel,parameters,x,y",
    [
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": jnp.log(1),
                }
            ),
            jnp.array(6.0),
            jnp.array(6.0),
        ],
    ],
)
def test_missing_kernel_parameter(
    kernel: StandardKernel,
    parameters: FrozenDict,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    with pytest.raises(KeyError):
        kernel.calculate_kernel(parameters=parameters, x=x, y=y)
