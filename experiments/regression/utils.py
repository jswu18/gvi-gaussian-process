from typing import Tuple

import jax
from jax import numpy as jnp
from sklearn.model_selection import train_test_split

from experiments.data import ExperimentData
from experiments.utils import calculate_inducing_points
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


def split_train_test_data_intervals(
    subkey: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split data into test and train data by randomly selecting intervals.

    Args:
        subkey: The random key.
        x: The x data.
        y: The y data.
        number_of_test_intervals: The number of chunks in the test data.
        total_number_of_intervals: The total number of intervals.

    Returns:
        The x and y data with the chunks removed.
    """
    total_number_of_intervals = max(
        total_number_of_intervals, number_of_test_intervals + 5
    )
    interval_size = x.shape[0] // total_number_of_intervals
    test_interval_indices = jax.random.choice(
        subkey,
        total_number_of_intervals - 4,
        shape=(number_of_test_intervals,),
        replace=False,
    )
    test_interval_indices = test_interval_indices + 2
    x_train = jnp.concatenate(
        [
            x[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval not in test_interval_indices
        ]
    )
    y_train = jnp.concatenate(
        [
            y[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval not in test_interval_indices
        ]
    )
    x_test = jnp.concatenate(
        [
            x[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval in test_interval_indices
        ]
    )
    y_test = jnp.concatenate(
        [
            y[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval in test_interval_indices
        ]
    )
    return x_train, y_train, x_test, y_test


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
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=1 - train_data_percentage,
        random_state=int(jnp.sum(subkey)),
    )
    return x_train, y_train, x_test, y_test, x_validation, y_validation


def set_up_regression_experiment(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    number_of_inducing_points: int,
    train_data_percentage: float,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> ExperimentData:
    key, subkey = jax.random.split(key)
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
