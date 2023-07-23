from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.means.base import MeanBase, MeanBaseParameters
from src.means.stochastic_variational_mean import (
    StochasticVariationalMean,
    StochasticVariationalMeanParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


class MockMeanParameters(MeanBaseParameters):
    pass


class MockMean(MeanBase):
    Parameters = MockMeanParameters

    def __init__(
        self,
        number_output_dimensions: int = 1,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        super().__init__(
            number_output_dimensions=number_output_dimensions,
            preprocess_function=preprocess_function,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockMeanParameters:
        return MockMeanParameters()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MockMeanParameters:
        return MockMeanParameters()

    def _predict(
        self,
        x: jnp.ndarray,
        parameters: MockMeanParameters = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0], self.number_output_dimensions))


class MockStochasticVariationalMeanParameter(StochasticVariationalMeanParameters):
    pass


class MockStochasticVariationalMean(StochasticVariationalMean):
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockStochasticVariationalMeanParameter:
        return MockStochasticVariationalMeanParameter()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MockStochasticVariationalMeanParameter:
        return MockStochasticVariationalMeanParameter()

    def _predict(
        self,
        x: jnp.ndarray,
        parameters: MockStochasticVariationalMeanParameter = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0]))
