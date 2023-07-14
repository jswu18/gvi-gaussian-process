import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernels import ReferenceKernelMock, ReferenceKernelParametersMock
from mockers.mean_functions import (
    ReferenceMeanFunctionMock,
    ReferenceMeanFunctionParametersMock,
)
from src.gaussian_measures.reference_gaussian_measures import ReferenceGaussianMeasure
from src.parameters.gaussian_measures.reference_gaussian_measures import (
    ReferenceGaussianMeasureParameters,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_observation_noise,x,y,x_test,mean",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array([1.8333333333333335, 1.8333333333333335]),
        ],
    ],
)
def test_reference_gaussian_measure_mean(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    mean: jnp.ndarray,
):
    gm = ReferenceGaussianMeasure(
        x=x,
        y=y,
        mean_function=ReferenceMeanFunctionMock(),
        kernel=ReferenceKernelMock(),
    )
    parameters = ReferenceGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ReferenceMeanFunctionParametersMock(),
        kernel=ReferenceKernelParametersMock(),
    )
    assert jnp.array_equal(gm.calculate_mean(parameters, x=x_test), mean)


@pytest.mark.parametrize(
    "log_observation_noise,x,y,x_test,covariance",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array(
                [
                    [0.33333333333333326, 0.33333333333333326],
                    [0.33333333333333326, 0.33333333333333326],
                ]
            ),
        ],
    ],
)
def test_reference_gaussian_measure_covariance(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    covariance: jnp.ndarray,
):
    gm = ReferenceGaussianMeasure(
        x=x,
        y=y,
        mean_function=ReferenceMeanFunctionMock(),
        kernel=ReferenceKernelMock(),
    )
    parameters = ReferenceGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ReferenceMeanFunctionParametersMock(),
        kernel=ReferenceKernelParametersMock(),
    )
    assert jnp.array_equal(gm.calculate_covariance(parameters, x=x_test), covariance)


@pytest.mark.parametrize(
    "log_observation_noise,x,y,observation_noise",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            1.0,
        ],
    ],
)
def test_reference_gaussian_measure_observation_noise(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    observation_noise: float,
):
    gm = ReferenceGaussianMeasure(
        x=x,
        y=y,
        mean_function=ReferenceMeanFunctionMock(),
        kernel=ReferenceKernelMock(),
    )
    parameters = ReferenceGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ReferenceMeanFunctionParametersMock(),
        kernel=ReferenceKernelParametersMock(),
    )
    assert gm.calculate_observation_noise(parameters) == observation_noise


@pytest.mark.parametrize(
    "log_observation_noise,x,y,negative_expected_log_likelihood",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            3.017407877979574,
        ],
    ],
)
def test_reference_gaussian_measure_negative_expected_log_likelihood(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    negative_expected_log_likelihood: float,
):
    gm = ReferenceGaussianMeasure(
        x=x,
        y=y,
        mean_function=ReferenceMeanFunctionMock(),
        kernel=ReferenceKernelMock(),
    )
    parameters = ReferenceGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ReferenceMeanFunctionParametersMock(),
        kernel=ReferenceKernelParametersMock(),
    )
    assert (
        gm.compute_negative_expected_log_likelihood(parameters=parameters, x=x, y=y)
        == negative_expected_log_likelihood
    )
