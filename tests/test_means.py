from typing import Dict

import pytest
from jax import numpy as jnp
from jax.config import config

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from mockers.neural_network import MockNeuralNetwork
from src.means import ConstantMean, CustomMean, MultiOutputMean, SVGPMean

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
            jnp.zeros((2,)),
        ],
        [
            ConstantMean(
                number_output_dimensions=5,
            ),
            {
                "constant": jnp.ones((5,)),
            },
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((5, 2)),
        ],
        [
            ConstantMean(
                number_output_dimensions=5,
                preprocess_function=lambda x: x.reshape(x.shape[0], -1),
            ),
            {
                "constant": jnp.ones((5,)),
            },
            jnp.ones((2, 3, 3, 1)),
            jnp.ones((5, 2)),
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
            jnp.ones((2,)),
        ],
    ],
)
def test_nn_mean(
    x: jnp.ndarray,
    mean: float,
):
    neural_network = MockNeuralNetwork()
    nn_mean = CustomMean(
        mean_function=lambda parameters, x: neural_network.apply(parameters, x=x),
    )
    nn_parameters = nn_mean.Parameters.construct(custom=None)
    assert jnp.array_equal(nn_mean.predict(parameters=nn_parameters, x=x), mean)


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
            4 * jnp.ones((2,)),
        ],
    ],
)
def test_svgp_mean(
    parameters: Dict,
    inducing_points: jnp.ndarray,
    x: jnp.ndarray,
    mean: float,
):
    svgp_mean = SVGPMean(
        regulariser_mean_parameters=MockMeanParameters(),
        regulariser_mean=MockMean(),
        regulariser_kernel_parameters=MockKernelParameters(),
        regulariser_kernel=MockKernel(),
        inducing_points=inducing_points,
    )
    assert jnp.array_equal(
        svgp_mean.predict(parameters=svgp_mean.generate_parameters(parameters), x=x),
        mean,
    )


@pytest.mark.parametrize(
    "number_of_outputs,x,mean",
    [
        [
            4,
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.ones((4, 2)),
        ],
    ],
)
def test_multi_output_mean(
    number_of_outputs: int,
    x: jnp.ndarray,
    mean: float,
):
    multi_output_mean = MultiOutputMean(
        means=[MockMean()] * number_of_outputs,
    )
    assert jnp.array_equal(
        multi_output_mean.predict(
            parameters=multi_output_mean.Parameters(
                means=[MockMeanParameters()] * number_of_outputs
            ),
            x=x,
        ),
        mean,
    )
