import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernels import MockKernel, MockKernelParameters
from mockers.means import MockMean, MockMeanParameters
from src.distributions import Multinomial
from src.gps import ApproximateGPClassification, GPClassification
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "number_of_classes,log_observation_noise,x,y,x_test,probabilities",
    [
        [
            4,
            jnp.log(jnp.array([1, 0.5, 0.2, 0.9])),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [0.5, 0.1, 0.2, 0.2],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array(
                [
                    [
                        0.29579238035540445,
                        0.19449721942556136,
                        0.2159886961681648,
                        0.29372170534428094,
                    ],
                    [
                        0.29579238035540445,
                        0.19449721942556136,
                        0.2159886961681648,
                        0.29372170534428094,
                    ],
                ]
            ),
        ],
    ],
)
def test_exact_gp_classification_prediction(
    number_of_classes: int,
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    probabilities: jnp.ndarray,
):
    gp = GPClassification(
        x=x,
        y=y,
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    multinomial = Multinomial(**gp.predict_probability(parameters, x=x_test).dict())
    assert jnp.array_equal(multinomial.probabilities, probabilities)


@pytest.mark.parametrize(
    "number_of_classes,log_observation_noise,x_test,probabilities",
    [
        [
            4,
            jnp.log(jnp.array([1, 0.5, 0.2, 0.9])),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array(
                [
                    [
                        0.2691234800778074,
                        0.24267525339935925,
                        0.22395749569068032,
                        0.26424377083219325,
                    ],
                    [
                        0.2691234800778074,
                        0.24267525339935925,
                        0.22395749569068032,
                        0.26424377083219325,
                    ],
                ]
            ),
        ],
    ],
)
def test_approximate_gp_classification_prediction(
    number_of_classes: int,
    log_observation_noise: float,
    x_test: jnp.ndarray,
    probabilities: jnp.ndarray,
):
    gp = ApproximateGPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    multinomial = Multinomial(**gp.predict_probability(parameters, x=x_test).dict())
    assert jnp.array_equal(multinomial.probabilities, probabilities)
