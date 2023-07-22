import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernels import MockKernel, MockKernelParameters
from mockers.means import MockMean, MockMeanParameters
from src.distributions import Gaussian
from src.gps import ApproximateGPRegression, GPRegression

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
            jnp.array([[1.8333333333333335, 1.8333333333333335]]),
        ],
    ],
)
def test_exact_gp_regression_mean(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    mean: jnp.ndarray,
):
    gp = GPRegression(
        x=x,
        y=y,
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MockKernelParameters(),
    )
    gaussian = Gaussian(**gp.predict_probability(parameters, x=x_test).dict())
    assert jnp.array_equal(gaussian.mean, mean)


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
def test_approximate_gp_regression_mean(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    mean: jnp.ndarray,
):
    gp = ApproximateGPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MockKernelParameters(),
    )
    gaussian = Gaussian(**gp.predict_probability(parameters, x=x_test).dict())
    assert jnp.array_equal(gaussian.mean, mean)


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
def test_gp_regression_posterior_covariance(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    covariance: jnp.ndarray,
):
    gp = GPRegression(
        x=x,
        y=y,
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MockKernelParameters(),
    )
    _, gp_covariance = gp.calculate_posterior(
        parameters,
        x_train=x,
        y_train=y,
        x=x_test,
    )
    assert jnp.array_equal(gp_covariance, covariance)


@pytest.mark.parametrize(
    "x,y,x_test,covariance",
    [
        [
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
                [0.33333333333333326, 0.33333333333333326],
            ),
        ],
    ],
)
def test_exact_gp_regression_prediction_covariance(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    covariance: jnp.ndarray,
):
    gp = GPRegression(
        x=x,
        y=y,
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=jnp.log(1.0),
        mean=MockMeanParameters(),
        kernel=MockKernelParameters(),
    )
    gaussian = Gaussian(
        **gp.predict_probability(
            parameters,
            x=x_test,
        ).dict()
    )
    assert jnp.array_equal(gaussian.covariance, covariance)


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
                [[2, 2]],
            ),
        ],
    ],
)
def test_approximate_gp_regression_prediction_covariance(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
    covariance: jnp.ndarray,
):
    gp = ApproximateGPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
    )
    parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MockKernelParameters(),
    )
    gaussian = Gaussian(
        **gp.predict_probability(
            parameters,
            x=x_test,
        ).dict()
    )
    assert jnp.array_equal(gaussian.covariance, covariance)