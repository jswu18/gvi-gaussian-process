import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.gaussian_measures import GaussianMeasureMock, GaussianMeasureParametersMock
from mockers.kernels import (
    ApproximateKernelMock,
    ApproximateKernelParametersMock,
    ReferenceKernelMock,
    ReferenceKernelParametersMock,
)
from mockers.mean_functions import (
    ApproximateMeanFunctionMock,
    ApproximateMeanFunctionParametersMock,
    ReferenceMeanFunctionMock,
    ReferenceMeanFunctionParametersMock,
)
from src.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasure,
)
from src.parameters.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasureParameters,
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
            jnp.array([1, 1]),
        ],
    ],
)
def test_approximate_gaussian_measure_mean(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    mean: jnp.ndarray,
):
    gm = ApproximateGaussianMeasure(
        x=x,
        y=y,
        mean_function=ApproximateMeanFunctionMock(
            reference_mean_function=ReferenceMeanFunctionMock(),
            reference_mean_function_parameters=ReferenceMeanFunctionParametersMock(),
        ),
        kernel=ApproximateKernelMock(
            reference_kernel=ReferenceKernelMock(),
            reference_kernel_parameters=ReferenceKernelParametersMock(),
        ),
        reference_gaussian_measure=GaussianMeasureMock(),
        reference_gaussian_measure_parameters=GaussianMeasureParametersMock(),
    )
    parameters = ApproximateGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ApproximateMeanFunctionParametersMock(),
        kernel=ApproximateKernelParametersMock(),
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
                    [1, 1],
                    [1, 1],
                ]
            ),
        ],
    ],
)
def test_approximate_gaussian_measure_covariance(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    covariance: jnp.ndarray,
):
    gm = ApproximateGaussianMeasure(
        x=x,
        y=y,
        mean_function=ApproximateMeanFunctionMock(
            reference_mean_function=ReferenceMeanFunctionMock(),
            reference_mean_function_parameters=ReferenceMeanFunctionParametersMock(),
        ),
        kernel=ApproximateKernelMock(
            reference_kernel=ReferenceKernelMock(),
            reference_kernel_parameters=ReferenceKernelParametersMock(),
        ),
        reference_gaussian_measure=GaussianMeasureMock(),
        reference_gaussian_measure_parameters=GaussianMeasureParametersMock(),
    )
    parameters = ApproximateGaussianMeasureParameters(
        log_observation_noise=log_observation_noise,
        mean_function=ApproximateMeanFunctionParametersMock(),
        kernel=ApproximateKernelParametersMock(),
    )
    assert jnp.array_equal(gm.calculate_covariance(parameters, x=x_test), covariance)
