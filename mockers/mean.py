from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.means.base import MeanBase, MeanBaseParameters
from src.means.svgp_mean import SVGPMean, SVGPMeanParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.utils.custom_types import PRNGKey


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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockMeanParameters:
        return MockMeanParameters()

    def _predict(
        self,
        x: jnp.ndarray,
        parameters: MockMeanParameters = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0], self.number_output_dimensions))


class MockSVGPMeanParameter(SVGPMeanParameters):
    pass


class MockSVGPMean(SVGPMean):
    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockSVGPMeanParameter:
        return MockSVGPMeanParameter()

    def _predict(
        self,
        x: jnp.ndarray,
        parameters: MockSVGPMeanParameter = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0]))
