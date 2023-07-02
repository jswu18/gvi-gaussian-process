from typing import Dict

import pytest
from jax import numpy as jnp
from jax.config import config

from mockers.mean_functions import NeuralNetworkMock
from src.mean_functions.mean_functions import (
    ConstantFunction,
    MeanFunction,
    NeuralNetworkMeanFunction,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "mean_function,parameters,x,mean",
    [
        [
            ConstantFunction(),
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
    ],
)
def test_reference_mean_functions(
    mean_function: MeanFunction,
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
            jnp.array([1, 1]),
        ],
    ],
)
def test_nn_mean_functions(
    x: jnp.ndarray,
    mean: float,
):
    nn_mean_function = NeuralNetworkMeanFunction(
        neural_network=NeuralNetworkMock(),
    )
    assert jnp.array_equal(
        nn_mean_function.predict(
            nn_mean_function.generate_parameters({"neural_network": None}), x=x
        ),
        mean,
    )
