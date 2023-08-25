import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.kernel import MockKernel, MockKernelParameters
from mockers.mean import MockMean, MockMeanParameters
from src.gps import ApproximateGPClassification, GPClassification
from src.kernels import MultiOutputKernel, MultiOutputKernelParameters
from src.regularisations import MultinomialWassersteinRegularisation
from src.regularisations.schemas import RegularisationMode

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x_train,y_train,x,multinomial_wasserstein_regularisation",
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
def test_zero_multinomial_wasserstein_gp_classification(
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    multinomial_wasserstein_regularisation: float,
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
    multinomial_wasserstein = MultinomialWassersteinRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
        mode=RegularisationMode.posterior,
    )
    assert jnp.isclose(
        multinomial_wasserstein.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        multinomial_wasserstein_regularisation,
    )


@pytest.mark.parametrize(
    "power,log_observation_noise,number_of_classes,x,multinomial_wasserstein_regularisation",
    [
        [
            4,
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
def test_zero_multinomial_wasserstein_approximate_gp_classification(
    power: int,
    log_observation_noise: float,
    number_of_classes,
    x: jnp.ndarray,
    multinomial_wasserstein_regularisation: float,
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
    multinomial_wasserstein = MultinomialWassersteinRegularisation(
        gp=gp,
        regulariser=gp,
        regulariser_parameters=gp_parameters,
        power=power,
        mode=RegularisationMode.posterior,
    )

    assert jnp.isclose(
        multinomial_wasserstein.calculate_regularisation(
            parameters=gp_parameters,
            x=x,
        ),
        multinomial_wasserstein_regularisation,
    )


@pytest.mark.parametrize(
    "log_observation_noise,number_of_classes,x_train,y_train,x,multinomial_wasserstein_regularisation",
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
            0.12691447,
        ],
    ],
)
def test_multinomial_wasserstein_gp_classification(
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    multinomial_wasserstein_regularisation: float,
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
    multinomial_wasserstein = MultinomialWassersteinRegularisation(
        gp=gp,
        regulariser=regulariser,
        regulariser_parameters=parameters,
        mode=RegularisationMode.posterior,
    )
    assert jnp.isclose(
        multinomial_wasserstein.calculate_regularisation(
            parameters=parameters,
            x=x,
        ),
        multinomial_wasserstein_regularisation,
    )


@pytest.mark.parametrize(
    "power,log_observation_noise,number_of_classes,x_train,y_train,x,multinomial_wasserstein_regularisation",
    [
        [
            5,
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
            0.10454125,
        ],
        [
            1,
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
            0.19894639,
        ],
    ],
)
def test_multinomial_wasserstein_gp_classification(
    power: int,
    log_observation_noise: float,
    number_of_classes,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x: jnp.ndarray,
    multinomial_wasserstein_regularisation: float,
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
    multinomial_wasserstein = MultinomialWassersteinRegularisation(
        gp=gp,
        regulariser=regulariser,
        regulariser_parameters=parameters,
        power=power,
        mode=RegularisationMode.posterior,
    )
    assert jnp.isclose(
        multinomial_wasserstein.calculate_regularisation(
            parameters=parameters,
            x=x,
        ),
        multinomial_wasserstein_regularisation,
    )
