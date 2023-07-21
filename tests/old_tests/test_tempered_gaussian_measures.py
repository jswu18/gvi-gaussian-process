import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.gaussian_measures import GaussianMeasureMock, GaussianMeasureParametersMock
from src.gaussian_measures.gaussian_measures import (
    TemperedGaussianMeasure,
    TemperedGaussianMeasureParameters,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_tempering_factor,x,y,x_test",
    [
        [
            jnp.log(0.5),
            jnp.ones((5, 3)),
            jnp.ones((2,)),
            jnp.ones((2, 3)),
        ],
        [
            jnp.log(50),
            jnp.zeros((5, 3)),
            jnp.ones((2,)),
            jnp.ones((2, 3)),
        ],
    ],
)
def test_tempered_gaussian_measure_mean(
    log_tempering_factor: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
):
    gaussian_measure = GaussianMeasureMock()
    gaussian_measure_parameters = GaussianMeasureParametersMock()
    tempered_gaussian_measure = TemperedGaussianMeasure(
        x=x,
        y=y,
        gaussian_measure=gaussian_measure,
        gaussian_measure_parameters=gaussian_measure_parameters,
    )
    tempered_gaussian_measure_parameters = TemperedGaussianMeasureParameters(
        log_tempering_factor=log_tempering_factor
    )

    assert jnp.array_equal(
        tempered_gaussian_measure.calculate_mean(
            parameters=tempered_gaussian_measure_parameters,
            x=x_test,
        ),
        gaussian_measure.calculate_mean(
            parameters=gaussian_measure_parameters,
            x=x_test,
        ),
    )


@pytest.mark.parametrize(
    "log_tempering_factor,x,y,x_test",
    [
        [
            jnp.log(0.5),
            jnp.ones((5, 3)),
            jnp.ones((2,)),
            jnp.ones((2, 3)),
        ],
        [
            jnp.log(50),
            jnp.zeros((5, 3)),
            jnp.ones((2,)),
            jnp.ones((2, 3)),
        ],
    ],
)
def test_tempered_gaussian_measure_covariance(
    log_tempering_factor: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_test: jnp.ndarray,
):
    gaussian_measure = GaussianMeasureMock()
    gaussian_measure_parameters = GaussianMeasureParametersMock()
    tempered_gaussian_measure = TemperedGaussianMeasure(
        x=x,
        y=y,
        gaussian_measure=gaussian_measure,
        gaussian_measure_parameters=gaussian_measure_parameters,
    )
    tempered_gaussian_measure_parameters = TemperedGaussianMeasureParameters(
        log_tempering_factor=log_tempering_factor
    )

    assert jnp.array_equal(
        tempered_gaussian_measure.calculate_covariance(
            parameters=tempered_gaussian_measure_parameters,
            x=x_test,
        ),
        jnp.exp(log_tempering_factor)
        * gaussian_measure.calculate_covariance(
            parameters=gaussian_measure_parameters,
            x=x_test,
        ),
    )
