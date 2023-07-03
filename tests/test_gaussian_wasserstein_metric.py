import warnings

import jax.numpy as jnp
import pytest
from jax.config import config

from mockers.gaussian_measures import GaussianMeasureMock, GaussianMeasureParametersMock
from mockers.kernels import ReferenceKernelMock, calculate_reference_gram_eye_mock
from src.gaussian_wasserstein_metric import (
    compute_gaussian_wasserstein_metric,
    compute_gaussian_wasserstein_metric_from_grams,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "x_train,gaussian_wasserstein_metric",
    [
        [
            jnp.ones((3, 3)),
            -1.999999810209374e-08,
        ],
        [
            jnp.ones((5, 3)),
            -1.999999899027216e-08,
        ],
    ],
)
def test_gwi_same_gaussian_measure_symmetric(
    x_train: jnp.ndarray,
    gaussian_wasserstein_metric: float,
):
    assert (
        compute_gaussian_wasserstein_metric(
            p=GaussianMeasureMock(),
            q=GaussianMeasureMock(),
            p_parameters=GaussianMeasureParametersMock(),
            q_parameters=GaussianMeasureParametersMock(),
            x_train=x_train,
            x_batch=x_train,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=True,
        )
        == gaussian_wasserstein_metric
    )


@pytest.mark.parametrize(
    "x_train,gaussian_wasserstein_metric",
    [
        [
            jnp.ones((3, 3)),
            -0.00023094343795992955,
        ],
        [
            jnp.ones((5, 3)),
            -0.0003577728757098164,
        ],
    ],
)
def test_gwi_same_gaussian_measure_non_symmetric(
    x_train: jnp.ndarray,
    gaussian_wasserstein_metric: float,
):
    warnings.filterwarnings("ignore")
    assert (
        compute_gaussian_wasserstein_metric(
            p=GaussianMeasureMock(),
            q=GaussianMeasureMock(),
            p_parameters=GaussianMeasureParametersMock(),
            q_parameters=GaussianMeasureParametersMock(),
            x_train=x_train,
            x_batch=x_train,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=False,
        )
        == gaussian_wasserstein_metric
    )


@pytest.mark.parametrize(
    "x_train,p_kernel_scale,q_kernel_scale,gaussian_wasserstein_metric",
    [
        [
            jnp.ones((3, 3)),
            1e2,
            1e5,
            93775.44461641769,
        ],
        [
            jnp.ones((5, 3)),
            1e-2,
            1e3,
            993.6854446164177,
        ],
    ],
)
def test_gwi_different_gaussian_measure_symmetric(
    x_train: jnp.ndarray,
    p_kernel_scale: float,
    q_kernel_scale: float,
    gaussian_wasserstein_metric: float,
):
    p = GaussianMeasureMock()
    p.kernel = ReferenceKernelMock(
        kernel_func=lambda x, y: calculate_reference_gram_eye_mock(x, y)
        * p_kernel_scale
    )
    q = GaussianMeasureMock()
    q.kernel = ReferenceKernelMock(
        kernel_func=lambda x, y: calculate_reference_gram_eye_mock(x, y)
        * q_kernel_scale
    )
    assert (
        compute_gaussian_wasserstein_metric(
            p=p,
            q=q,
            p_parameters=GaussianMeasureParametersMock(),
            q_parameters=GaussianMeasureParametersMock(),
            x_train=x_train,
            x_batch=x_train,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=True,
        )
        == gaussian_wasserstein_metric
    )


@pytest.mark.parametrize(
    "x_train,p_kernel_scale,q_kernel_scale,gaussian_wasserstein_metric",
    [
        [
            jnp.ones((3, 3)),
            1e2,
            1e5,
            93775.44464804046,
        ],
        [
            jnp.ones((5, 3)),
            1e-2,
            1e3,
            993.6854446480404,
        ],
    ],
)
def test_gwi_different_gaussian_measure_non_symmetric(
    x_train: jnp.ndarray,
    p_kernel_scale: float,
    q_kernel_scale: float,
    gaussian_wasserstein_metric: float,
):
    warnings.filterwarnings("ignore")
    p = GaussianMeasureMock()
    p.kernel = ReferenceKernelMock(
        kernel_func=lambda x, y: calculate_reference_gram_eye_mock(x, y)
        * p_kernel_scale
    )
    q = GaussianMeasureMock()
    q.kernel = ReferenceKernelMock(
        kernel_func=lambda x, y: calculate_reference_gram_eye_mock(x, y)
        * q_kernel_scale
    )
    assert (
        compute_gaussian_wasserstein_metric(
            p=p,
            q=q,
            p_parameters=GaussianMeasureParametersMock(),
            q_parameters=GaussianMeasureParametersMock(),
            x_train=x_train,
            x_batch=x_train,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=False,
        )
        == gaussian_wasserstein_metric
    )


@pytest.mark.parametrize(
    """
    mean_train_p,covariance_train_p,
    mean_train_q,covariance_train_q,
    gram_batch_train_p,gram_batch_train_q,
    gaussian_wasserstein_metric
    """,
    [
        [
            jnp.ones((3,)),
            jnp.eye(3),
            jnp.ones((3,)),
            jnp.eye(3),
            jnp.ones((3, 3)),
            jnp.ones((3, 3)),
            -1.999999810209374e-08,
        ],
        [
            -jnp.ones((3,)),
            jnp.eye(3),
            3.7 * jnp.ones((3,)),
            0.4 * jnp.eye(3),
            jnp.ones((3, 3)),
            2.1 * jnp.ones((3, 3)),
            20.591724621779363,
        ],
    ],
)
def test_gwi_gaussian_measure_from_grams_symmetric(
    mean_train_p: jnp.ndarray,
    covariance_train_p: jnp.ndarray,
    mean_train_q: jnp.ndarray,
    covariance_train_q: jnp.ndarray,
    gram_batch_train_p: jnp.ndarray,
    gram_batch_train_q: jnp.ndarray,
    gaussian_wasserstein_metric: float,
):
    assert (
        compute_gaussian_wasserstein_metric_from_grams(
            mean_train_p=mean_train_p,
            covariance_train_p=covariance_train_p,
            mean_train_q=mean_train_q,
            covariance_train_q=covariance_train_q,
            gram_batch_train_p=gram_batch_train_p,
            gram_batch_train_q=gram_batch_train_q,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=True,
        )
        == gaussian_wasserstein_metric
    )


@pytest.mark.parametrize(
    """
    mean_train_p,covariance_train_p,
    mean_train_q,covariance_train_q,
    gram_batch_train_p,gram_batch_train_q,
    gaussian_wasserstein_metric
    """,
    [
        [
            jnp.ones((3,)),
            jnp.eye(3),
            jnp.ones((3,)),
            jnp.eye(3),
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            -0.00014142635423741723,
        ],
        [
            -jnp.ones((3,)),
            jnp.eye(3),
            3.7 * jnp.ones((3,)),
            0.4 * jnp.eye(3),
            jnp.ones((2, 3)),
            2.1 * jnp.ones((2, 3)),
            20.591519704503572,
        ],
    ],
)
def test_gwi_gaussian_measure_from_grams_non_symmetric(
    mean_train_p: jnp.ndarray,
    covariance_train_p: jnp.ndarray,
    mean_train_q: jnp.ndarray,
    covariance_train_q: jnp.ndarray,
    gram_batch_train_p: jnp.ndarray,
    gram_batch_train_q: jnp.ndarray,
    gaussian_wasserstein_metric: float,
):
    warnings.filterwarnings("ignore")
    assert (
        compute_gaussian_wasserstein_metric_from_grams(
            mean_train_p=mean_train_p,
            covariance_train_p=covariance_train_p,
            mean_train_q=mean_train_q,
            covariance_train_q=covariance_train_q,
            gram_batch_train_p=gram_batch_train_p,
            gram_batch_train_q=gram_batch_train_q,
            eigenvalue_regularisation=1e-8,
            is_eigenvalue_regularisation_absolute_scale=False,
            use_symmetric_matrix_eigendecomposition=False,
        )
        == gaussian_wasserstein_metric
    )
