from typing import Dict

import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernels import ReferenceKernelMock, ReferenceKernelParametersMock
from mockers.mean_functions import (
    NeuralNetworkMock,
    ReferenceMeanFunctionMock,
    ReferenceMeanFunctionParametersMock,
)
from src.mean_functions.approximate_mean_functions import (
    StochasticVariationalGaussianProcessMeanFunction,
)
from src.mean_functions.mean_functions import NeuralNetworkMeanFunction

config.update("jax_enable_x64", True)


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
            jnp.array([4, 4]),
        ],
    ],
)
def test_svgp_mean_functions(
    parameters: Dict,
    inducing_points: jnp.ndarray,
    x: jnp.ndarray,
    mean: float,
):
    svgp_mean_function = StochasticVariationalGaussianProcessMeanFunction(
        reference_mean_function_parameters=ReferenceMeanFunctionParametersMock(),
        reference_mean_function=ReferenceMeanFunctionMock(),
        reference_kernel_function_parameters=ReferenceKernelParametersMock(),
        reference_kernel=ReferenceKernelMock(),
        inducing_points=inducing_points,
    )
    assert jnp.array_equal(
        svgp_mean_function.predict(
            svgp_mean_function.generate_parameters(parameters), x=x
        ),
        mean,
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
            jnp.array([1, 1]),
        ],
    ],
)
def test_nn_mean_functions(
    x: jnp.ndarray,
    mean: float,
):
    nn_mean_function = NeuralNetworkMeanFunction(
        reference_mean_function_parameters=ReferenceMeanFunctionParametersMock(),
        reference_mean_function=ReferenceMeanFunctionMock(),
        neural_network=NeuralNetworkMock(),
    )
    assert jnp.array_equal(
        nn_mean_function.predict(
            nn_mean_function.generate_parameters({"neural_network": None}), x=x
        ),
        mean,
    )
