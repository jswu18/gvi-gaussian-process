from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.custom_types import PRNGKey


def split_train_test_data(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    train_data_percentage: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split data into test and train data by randomly selecting points.

    Args:
        key: The random key.
        x: The x data.
        y: The y data.
        train_data_percentage: The percentage of data to use for training.

    Returns:
        Train and test data.
    """
    number_of_training_points = int(x.shape[0] * train_data_percentage)

    train_idx = jax.random.choice(
        key, x.shape[0], shape=(number_of_training_points,), replace=False
    )
    train_mask = np.zeros(x.shape[0]).astype(bool)
    train_mask[train_idx] = True
    x_train = x[train_mask, ...]
    y_train = y[train_mask]
    x_test = x[~train_mask, ...]
    y_test = y[~train_mask]

    return x_train, y_train, x_test, y_test


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
    interval_size = x.shape[0] // total_number_of_intervals
    test_interval_indices = jax.random.choice(
        subkey,
        total_number_of_intervals - 2,
        shape=(number_of_test_intervals,),
        replace=False,
    )
    test_interval_indices = test_interval_indices + 1
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
