import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernel import (
    MockKernel,
    MockKernelParameters,
    calculate_reference_gram_eye_mock,
)
from mockers.mean import MockMean, MockMeanParameters
from src.distributions import Multinomial
from src.gps import ApproximateGPClassification, GPClassification
from src.kernels import (
    MultiOutputKernel,
    MultiOutputKernelParameters,
    TemperedKernel,
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
    "number_of_classes,x_test,probabilities",
    [
        [
            4,
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ),
        ],
    ],
)
def test_approximate_gp_classification_prediction(
    number_of_classes: int,
    x_test: jnp.ndarray,
    probabilities: jnp.ndarray,
):
    gp = ApproximateGPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
    )
    parameters = gp.Parameters(
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    multinomial = Multinomial(**gp.predict_probability(parameters, x=x_test).dict())
    assert jnp.allclose(multinomial.probabilities, probabilities)


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
                    [0.25269439, 0.19932216, 0.22324345, 0.32474002],
                    [0.25269439, 0.19932216, 0.22324345, 0.32474002],
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
        kernel=MultiOutputKernel(
            kernels=[MockKernel(kernel_func=calculate_reference_gram_eye_mock)]
            * number_of_classes
        ),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    tempered_gp = type(gp)(
        mean=gp.mean,
        kernel=TemperedKernel(
            base_kernel=gp.kernel,
            base_kernel_parameters=parameters.kernel,
            number_output_dimensions=gp.kernel.number_output_dimensions,
        ),
        x=x,
        y=y,
    )
    tempered_gp_parameters = tempered_gp.Parameters(
        log_observation_noise=parameters.log_observation_noise,
        mean=parameters.mean,
        kernel=TemperedKernelParameters(log_tempering_factor=log_tempering_factor),
    )
    tempered_gp.kernel._calculate_gram(tempered_gp_parameters.kernel, x1=x, x2=x)
    multinomial = Multinomial(
        **tempered_gp.predict_probability(tempered_gp_parameters, x=x_test).dict()
    )
    assert jnp.allclose(multinomial.probabilities, probabilities)


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
                    [0.1690632, 0.20674086, 0.29816303, 0.32603688],
                    [0.1690632, 0.20674086, 0.29816303, 0.32603688],
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
    tempered_gp = type(gp)(
        mean=gp.mean,
        kernel=TemperedKernel(
            base_kernel=gp.kernel,
            base_kernel_parameters=parameters.kernel,
            number_output_dimensions=gp.kernel.number_output_dimensions,
        ),
    )
    tempered_gp_parameters = tempered_gp.Parameters(
        log_observation_noise=parameters.log_observation_noise,
        mean=parameters.mean,
        kernel=TemperedKernelParameters(log_tempering_factor=log_tempering_factor),
    )
    multinomial = Multinomial(
        **tempered_gp.predict_probability(tempered_gp_parameters, x=x_test).dict()
    )
    assert jnp.allclose(multinomial.probabilities, probabilities)
