import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from src.distributions import Multinomial
from src.gps import (
    ApproximateGPClassification,
    GPClassification,
    TemperedGP,
    TemperedGPParameters,
)
from src.kernels import (
    MultiOutputKernel,
    MultiOutputKernelParameters,
    TemperedKernelParameters,
)

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


@pytest.mark.parametrize(
    "log_tempering_factor,number_of_classes,log_observation_noise,x,y,x_test,probabilities",
    [
        [
            jnp.log(jnp.array([0.5, 1.0, 3.0, 4.0])),
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
                        0.22861418433856942,
                        0.18940287159753366,
                        0.22580482612573324,
                        0.3561781566207092,
                    ],
                    [
                        0.22861418433856942,
                        0.18940287159753366,
                        0.22580482612573324,
                        0.3561781566207092,
                    ],
                ]
            ),
        ],
    ],
)
def test_tempered_exact_gp_classification_prediction(
    log_tempering_factor: jnp.ndarray,
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
    tempered_gp = TemperedGP(
        base_gp=gp,
        base_gp_parameters=parameters,
    )
    tempered_gp_parameters = TemperedGPParameters(
        kernel=TemperedKernelParameters(
            log_tempering_factor=log_tempering_factor,
        )
    )
    multinomial = Multinomial(
        **tempered_gp.predict_probability(tempered_gp_parameters, x=x_test).dict()
    )
    assert jnp.array_equal(multinomial.probabilities, probabilities)


@pytest.mark.parametrize(
    "log_tempering_factor,number_of_classes,log_observation_noise,x_test,probabilities",
    [
        [
            jnp.log(jnp.array([0.5, 1.0, 3.0, 4.0])),
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
                        0.20811607036034108,
                        0.20811607036034108,
                        0.27109154535384067,
                        0.3126763214158399,
                    ],
                    [
                        0.20811607036034108,
                        0.20811607036034108,
                        0.27109154535384067,
                        0.3126763214158399,
                    ],
                ]
            ),
        ],
    ],
)
def test_tempered_approximate_gp_classification_prediction(
    log_tempering_factor: jnp.ndarray,
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
    tempered_gp = TemperedGP(
        base_gp=gp,
        base_gp_parameters=parameters,
    )
    tempered_gp_parameters = TemperedGPParameters(
        kernel=TemperedKernelParameters(
            log_tempering_factor=log_tempering_factor,
        )
    )
    multinomial = Multinomial(
        **tempered_gp.predict_probability(tempered_gp_parameters, x=x_test).dict()
    )
    assert jnp.array_equal(multinomial.probabilities, probabilities)
