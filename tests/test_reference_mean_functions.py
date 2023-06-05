from typing import Dict

import jax.numpy as jnp
import pytest
from jax.config import config

from src.mean_functions.mean_functions import MeanFunction
from src.mean_functions.reference_mean_functions import ConstantFunction

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
def test_standard_kernels(
    mean_function: MeanFunction,
    parameters: Dict,
    x: jnp.ndarray,
    mean: float,
):
    assert jnp.array_equal(
        mean_function.predict(mean_function.generate_parameters(parameters), x=x), mean
    )
