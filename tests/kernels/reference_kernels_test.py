from typing import Callable, List

import jax.numpy as jnp
import pytest
from flax.core.frozen_dict import FrozenDict

from src.kernels.reference_kernels import ARDKernel, StandardKernel


@pytest.mark.parametrize(
    "kernel,parameters,x,y,k",
    [
        [
            ARDKernel(),
            FrozenDict(
                {
                    "log_scaling": jnp.array([-0.5, 0, 0.5]),
                    "log_lengthscales": jnp.array([-0.5, 0, 0.5]),
                }
            ),
            jnp.array(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                ]
            ),
            jnp.array(
                [
                    [1.5],
                    [2.5],
                    [3.5],
                ]
            ),
            2.7201838,
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
    ],
)
def test_standard_kernels(
    kernel: StandardKernel,
    parameters: FrozenDict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    k: float,
):
    assert kernel.kernel_func(parameters, x, y) == k
