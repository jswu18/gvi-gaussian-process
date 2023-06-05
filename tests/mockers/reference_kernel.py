from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.kernels import Kernel
from src.parameters.kernels.kernels import KernelParameters
from src.parameters.module import ModuleParameters

PRNGKey = Any  # pylint: disable=invalid-name


class ReferenceKernelParametersMock(KernelParameters):
    pass


class ReferenceKernelMock(Kernel):
    sigma_matrix = jnp.ones((5, 5))

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ReferenceKernelParametersMock:
        return ReferenceKernelParametersMock()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ReferenceKernelParametersMock:
        return ReferenceKernelParametersMock()

    def _calculate_gram(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        parameters: ReferenceKernelParametersMock = None,
    ) -> jnp.ndarray:
        return jnp.ones((x.shape[0], y.shape[0]))


def calculate_reference_gram_mock(
    x: jnp.ndarray,
    y: jnp.ndarray = None,
) -> jnp.ndarray:
    return jnp.ones((x.shape[0], y.shape[0]))
