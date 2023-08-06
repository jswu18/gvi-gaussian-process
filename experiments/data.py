from dataclasses import dataclass

import jax
from jax import numpy as jnp
from sklearn.model_selection import train_test_split

from src.utils.custom_types import PRNGKey


@dataclass
class Data:
    x: jnp.ndarray
    y: jnp.ndarray

    def __add__(self, other):
        return Data(
            x=jnp.concatenate([self.x, other.x]),
            y=jnp.concatenate([self.y, other.y]),
        )


@dataclass
class ExperimentData:
    full: Data
    train: Data
    test: Data
    validation: Data

    def __add__(self, other):
        return ExperimentData(
            full=self.full + other.full,
            train=self.train + other.train,
            test=self.test + other.test,
            validation=self.validation + other.validation,
        )


def set_up_experiment(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    train_data_percentage: float,
    test_data_percentage: float,
    validation_data_percentage: float,
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
        random_state=int(jnp.sum(subkey)) % (2**32 - 1),
    )

    key, subkey = jax.random.split(key)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_test_and_validation,
        y_test_and_validation,
        test_size=test_data_percentage
        / (test_data_percentage + validation_data_percentage),
        random_state=int(jnp.sum(subkey)) % (2**32 - 1),
    )
    experiment_data = ExperimentData(
        full=Data(x=x, y=y),
        train=Data(x=x_train, y=y_train),
        test=Data(x=x_test, y=y_test),
        validation=Data(x=x_validation, y=y_validation),
    )
    return experiment_data
