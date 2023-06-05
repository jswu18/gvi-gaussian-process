from typing import Any, Dict, Union

import flax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.mean_functions.mean_functions import MeanFunction
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters
from src.parameters.module import ModuleParameters

PRNGKey = Any  # pylint: disable=invalid-name


class ReferenceMeanFunctionParametersMock(MeanFunctionParameters):
    pass


class ReferenceMeanFunctionMock(MeanFunction):
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ReferenceMeanFunctionParametersMock:
        return ReferenceMeanFunctionParametersMock()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ReferenceMeanFunctionParametersMock:
        return ReferenceMeanFunctionParametersMock()

    def _predict(
        self,
        x: jnp.ndarray,
        parameters: ReferenceMeanFunctionParametersMock = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0]))


class NeuralNetworkMock(flax.linen.Module):
    def init(self, *args, **kwargs):
        return None

    def apply(self, variables, *args, **kwargs):
        if "x" in kwargs:
            return jnp.ones((kwargs["x"].shape[0]))
        else:
            return jnp.ones((args[0].shape[0]))
