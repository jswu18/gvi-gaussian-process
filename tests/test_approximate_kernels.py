from typing import Dict

import jax.numpy as jnp
import pytest
from jax.config import config
from mock import MagicMock

from mockers.kernels import (
    ReferenceKernelMock,
    ReferenceKernelParametersMock,
    calculate_reference_gram_mock,
)
from src.kernels.approximate_kernels import StochasticVariationalGaussianProcessKernel
from src.parameters.kernels.approximate_kernels import (
    StochasticVariationalGaussianProcessKernelParameters,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    """
    gram_inducing,
    gram_inducing_train,
    reference_gaussian_measure_observation_precision,
    diagonal_regularisation,
    is_diagonal_regularisation_absolute_scale,
    sigma_matrix
    """,
    [
        [
            jnp.ones((2, 2)),
            jnp.ones((2, 3)),
            1,
            1e-5,
            False,
            jnp.array([[0.2499975000249998, 0], [0, 12500.062499750054]]),
        ],
    ],
)
def test_calculate_sigma_matrix(
    gram_inducing: jnp.ndarray,
    gram_inducing_train: jnp.ndarray,
    reference_gaussian_measure_observation_precision: float,
    diagonal_regularisation: float,
    is_diagonal_regularisation_absolute_scale: bool,
    sigma_matrix: jnp.array,
):

    assert jnp.array_equal(
        StochasticVariationalGaussianProcessKernel._calculate_sigma_matrix(
            gram_inducing,
            gram_inducing_train,
            reference_gaussian_measure_observation_precision,
            diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale,
        ),
        sigma_matrix,
    )


@pytest.mark.parametrize(
    "parameters,x,y,gram",
    [
        [
            {},
            jnp.array([[1.0, 2.0, 3.0]]),
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            jnp.array([[12500.312502250054, 12500.312502250054]]),
        ],
    ],
)
def test_svgp_kernel_grams(
    parameters: Dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    gram: float,
):
    approximate_kernel = StochasticVariationalGaussianProcessKernel(
        reference_kernel_parameters=ReferenceKernelParametersMock(),
        log_observation_noise=jnp.log(1),
        reference_kernel=ReferenceKernelMock(),
        inducing_points=jnp.ones((2, 1)),
        training_points=jnp.ones((3, 1)),
    )
    approximate_kernel.calculate_reference_gram = MagicMock(
        side_effect=calculate_reference_gram_mock
    )
    assert jnp.array_equal(
        approximate_kernel.calculate_gram(
            StochasticVariationalGaussianProcessKernelParameters(), x=x, y=y
        ),
        gram,
    )
