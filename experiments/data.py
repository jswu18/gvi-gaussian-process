from dataclasses import dataclass

import jax
from jax import numpy as jnp

from experiments.regression.toy_curves import Curve
from experiments.regression.utils import (
    split_train_test_data,
    split_train_test_data_intervals,
)
from experiments.utils import calculate_inducing_points
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


@dataclass
class ExperimentData:
    x: jnp.ndarray
    y: jnp.ndarray
    x_train: jnp.ndarray
    y_train: jnp.ndarray
    x_inducing: jnp.ndarray
    y_inducing: jnp.ndarray
    x_test: jnp.ndarray
    y_test: jnp.ndarray
    x_validation: jnp.ndarray
    y_validation: jnp.ndarray


def set_up_experiment(
    key: PRNGKey,
    curve_function: Curve,
    x: jnp.ndarray,
    sigma_true: float,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    number_of_inducing_points: int,
    train_data_percentage: float,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> ExperimentData:
    key, subkey = jax.random.split(key)
    y = curve_function(key=subkey, x=x, sigma_true=sigma_true)
    key, subkey = jax.random.split(key)
    (
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation,
    ) = split_train_test_validation_data(
        key=subkey,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
        train_data_percentage=train_data_percentage,
    )
    key, subkey = jax.random.split(key)
    x_inducing, y_inducing = calculate_inducing_points(
        key=subkey,
        x=x_train,
        y=y_train,
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    experiment_data = ExperimentData(
        x=x,
        y=y,
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_inducing,
        y_inducing=y_inducing,
        y_test=y_test,
        x_test=x_test,
        x_validation=x_validation,
        y_validation=y_validation,
    )
    return experiment_data


def split_train_test_validation_data(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    train_data_percentage: float,
):
    key, subkey = jax.random.split(key)
    (
        x_train_validation,
        y_train_validation,
        x_test,
        y_test,
    ) = split_train_test_data_intervals(
        subkey=subkey,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    key, subkey = jax.random.split(key)
    x_train, y_train, x_validation, y_validation = split_train_test_data(
        key=subkey,
        x=x_train_validation,
        y=y_train_validation,
        train_data_percentage=train_data_percentage,
    )
    return x_train, y_train, x_test, y_test, x_validation, y_validation
