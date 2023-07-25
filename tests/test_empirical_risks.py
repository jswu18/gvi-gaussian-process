import jax.numpy as jnp
import pytest

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from src.empirical_risks import NegativeLogLikelihood
from src.gps import (
    ApproximateGPClassification,
    ApproximateGPRegression,
    GPClassification,
    GPRegression,
)
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters


@pytest.mark.parametrize(
    "log_observation_noise,x_train,y_train,x,y,negative_log_likelihood",
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
            jnp.ones((2,)),
            0.11574074,
        ],
    ],
)
def test_gp_regression_nll(
    log_observation_noise: float,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    negative_log_likelihood: float,
):
    gp = GPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
    )
    nll = NegativeLogLikelihood(
        gp=gp,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMean.Parameters(),
        kernel=MockKernel.Parameters(),
    )
    assert jnp.isclose(
        nll.calculate_empirical_risk(
            parameters=gp_parameters,
            x=x,
            y=y,
        ),
        negative_log_likelihood,
    )


@pytest.mark.parametrize(
    "log_observation_noise,x,y,negative_log_likelihood",
    [
        [
            jnp.log(1.0),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([1.2, 4.2]),
            5.14,
        ],
    ],
)
def test_approximate_gp_regression_nll(
    log_observation_noise: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    negative_log_likelihood: float,
):
    gp = ApproximateGPRegression(
        mean=MockMean(),
        kernel=MockKernel(),
    )
    nll = NegativeLogLikelihood(
        gp=gp,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMean.Parameters(),
        kernel=MockKernel.Parameters(),
    )
    assert jnp.isclose(
        nll.calculate_empirical_risk(
            parameters=gp_parameters,
            x=x,
            y=y,
        ),
        negative_log_likelihood,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x_train,y_train,x,y,negative_log_likelihood",
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
            jnp.array(
                [
                    [0.5, 0.1, 0.2, 0.2],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ),
            0.07816624,
        ],
    ],
)
def test_gp_classification_nll(
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    negative_log_likelihood: float,
):
    gp = GPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
        x=x_train,
        y=y_train,
    )
    nll = NegativeLogLikelihood(
        gp=gp,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    assert jnp.isclose(
        nll.calculate_empirical_risk(
            parameters=gp_parameters,
            x=x,
            y=y,
        ),
        negative_log_likelihood,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x,y,negative_log_likelihood",
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
            jnp.array(
                [
                    [0.5, 0.1, 0.2, 0.2],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ),
            0.4445,
        ],
    ],
)
def test_gp_approximate_classification_nll(
    log_observation_noise: float,
    number_of_classes,
    x: jnp.ndarray,
    y: jnp.ndarray,
    negative_log_likelihood: float,
):
    gp = ApproximateGPClassification(
        mean=MockMean(number_output_dimensions=number_of_classes),
        kernel=MultiOutputKernel(kernels=[MockKernel()] * number_of_classes),
    )
    nll = NegativeLogLikelihood(
        gp=gp,
    )
    gp_parameters = gp.Parameters(
        log_observation_noise=log_observation_noise,
        mean=MockMeanParameters(),
        kernel=MultiOutputKernelParameters(
            kernels=[MockKernelParameters()] * number_of_classes
        ),
    )
    assert jnp.isclose(
        nll.calculate_empirical_risk(
            parameters=gp_parameters,
            x=x,
            y=y,
        ),
        negative_log_likelihood,
    )
