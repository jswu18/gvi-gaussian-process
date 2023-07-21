from typing import Dict

import pytest
from jax import numpy as jnp
from jax.config import config

from mockers.kernels import MockKernel, MockKernelParameters
from mockers.mean_functions import MockMean, MockMeanParameters, MockNeuralNetwork
from src.means import ConstantMean, NeuralNetworkMean, StochasticVariationalMean

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "mean_function,parameters,x,mean",
    [
        [
            ConstantMean(),
            {
                "constant": 0,
            },
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.zeros((2, 1)),
        ],
        [
            ConstantMean(),
            {
                "constant": jnp.ones((5,)),
            },
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((2, 5)),
        ],
    ],
)
def test_constant_mean(
    mean_function: ConstantMean,
    parameters: Dict,
    x: jnp.ndarray,
    mean: float,
):
    assert jnp.array_equal(
        mean_function.predict(mean_function.generate_parameters(parameters), x=x), mean
    )


@pytest.mark.parametrize(
    "x,mean",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((2, 1)),
        ],
    ],
)
def test_nn_mean_functions(
    x: jnp.ndarray,
    mean: float,
):
    nn_mean = NeuralNetworkMean(
        neural_network=MockNeuralNetwork(),
    )
    assert jnp.array_equal(
        nn_mean.predict(nn_mean.generate_parameters({"neural_network": None}), x=x),
        mean,
    )


@pytest.mark.parametrize(
    "parameters,inducing_points,x,mean",
    [
        [
            {"weights": jnp.array([1, 2])},
            jnp.ones((2, 3)),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            4 * jnp.ones((2, 1)),
        ],
    ],
)
def test_svgp_mean_functions(
    parameters: Dict,
    inducing_points: jnp.ndarray,
    x: jnp.ndarray,
    mean: float,
):
    svgp_mean = StochasticVariationalMean(
        reference_mean_parameters=MockMeanParameters(),
        reference_mean=MockMean(),
        reference_kernel_parameters=MockKernelParameters(),
        reference_kernel=MockKernel(),
        inducing_points=inducing_points,
    )
    assert jnp.array_equal(
        svgp_mean.predict(parameters=svgp_mean.generate_parameters(parameters), x=x),
        mean,
    )
