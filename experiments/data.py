from dataclasses import dataclass
from typing import Tuple

import jax
import numpy as np
from jax import numpy as jnp
from sklearn.model_selection import train_test_split

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

    def __add__(self, other):
        return ExperimentData(
            x=jnp.concatenate([self.x, other.x]),
            y=jnp.concatenate([self.y, other.y]),
            x_train=jnp.concatenate([self.x_train, other.x_train]),
            y_train=jnp.concatenate([self.y_train, other.y_train]),
            x_inducing=jnp.concatenate([self.x_inducing, other.x_inducing]),
            y_inducing=jnp.concatenate([self.y_inducing, other.y_inducing]),
            x_test=jnp.concatenate([self.x_test, other.x_test]),
            y_test=jnp.concatenate([self.y_test, other.y_test]),
            x_validation=jnp.concatenate([self.x_validation, other.x_validation]),
            y_validation=jnp.concatenate([self.y_validation, other.y_validation]),
        )


def set_up_experiment(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_inducing_points: int,
    train_data_percentage: float,
    test_data_percentage: float,
    validation_data_percentage: float,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> ExperimentData:
    # adapted from https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
    key, subkey = jax.random.split(key)
    (
        x_train,
        x_test_and_validation,
        y_train,
        y_test_and_validation,
    ) = train_test_split(
        x,
        y,
        test_size=1 - train_data_percentage,
        random_state=int(jnp.sum(subkey)),
    )

    key, subkey = jax.random.split(key)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_test_and_validation,
        y_test_and_validation,
        test_size=test_data_percentage
        / (test_data_percentage + validation_data_percentage),
        random_state=int(jnp.sum(subkey)),
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
