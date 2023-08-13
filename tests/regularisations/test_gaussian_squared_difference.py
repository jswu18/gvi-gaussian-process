import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from src.gps import (
    ApproximateGPClassification,
    ApproximateGPRegression,
    GPClassification,
    GPRegression,
)
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters
from src.regularisations import GaussianSquaredDifferenceRegularisation

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_observation_noise,x_train,y_train,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0,
        ],
    ],
)
def test_zero_gaussian_squared_difference_gp_regression(
    log_observation_noise: float,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation,
):
    gp = GPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMean.Parameters(),
        kernel=MockKernel.Parameters(),
    )
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
    )
    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0,
        ],
    ],
)
def test_zero_gaussian_squared_difference_approximate_gp_regression(
    log_observation_noise: float,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation,
):
    gp = ApproximateGPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMean.Parameters(),
        kernel=MockKernel.Parameters(),
    )
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
    )

    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,x_train,y_train,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
                ]
            ),
            jnp.array([1.0, 1.5]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            1.13888889,
        ],
    ],
)
def test_gaussian_squared_difference_gp_regression(
    log_observation_noise: float,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation,
):
    regulariser = GPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
    )
    gp = ApproximateGPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMean.Parameters(),
        kernel=MockKernel.Parameters(),
    )
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=regulariser,
        regulariser_parameters=parameters,
    )
    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x_train,y_train,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(jnp.array([0.1, 0.2, 0.4, 1.8])),
            4,
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
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
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0,
        ],
    ],
)
def test_zero_gaussian_squared_difference_gp_classification(
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation: float,
):
    gp = GPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
        x=x_train,
        y=y_train,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
    )
    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(jnp.array([0.1, 0.2, 0.4, 1.8])),
            4,
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0,
        ],
    ],
)
def test_zero_gaussian_squared_difference_approximate_gp_classification(
    log_observation_noise: float,
    number_of_classes,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation: float,
):
    gp = ApproximateGPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
    )

    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x_train,y_train,x,gaussian_squared_difference_regularisation",
    [
        [
            jnp.log(jnp.array([0.1, 0.2, 0.4, 1.8])),
            4,
            jnp.array(
                [
                    [1.0, 3.0, 2.0],
                    [1.5, 1.5, 9.5],
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
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0.71837243,
        ],
    ],
)
def test_gaussian_squared_difference_gp_classification(
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    gaussian_squared_difference_regularisation: float,
):
    regulariser = GPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
        x=x_train,
        y=y_train,
    )
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
    gaussian_squared_difference = GaussianSquaredDifferenceRegularisation(
        gp=gp,
        regulariser=regulariser,
        regulariser_parameters=parameters,
    )
    assert jnp.isclose(
        gaussian_squared_difference.calculate_regularisation(
            parameters=parameters,
            x=x,
        ),
        gaussian_squared_difference_regularisation,
    )
