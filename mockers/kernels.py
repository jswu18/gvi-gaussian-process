from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.approximate_kernels import ApproximateKernel
from src.kernels.reference_kernels import ReferenceKernel
from src.parameters.kernels.approximate_kernels import ApproximateKernelParameters
from src.parameters.kernels.reference_kernels import (
    KernelParameters as ReferenceKernelParameters,
)

PRNGKey = Any  # pylint: disable=invalid-name


def calculate_reference_gram_mock(
    x: jnp.ndarray,
    y: jnp.ndarray = None,
) -> jnp.ndarray:
    return jnp.ones((x.shape[0], y.shape[0]))


def calculate_reference_gram_eye_mock(
    x: jnp.ndarray,
    y: jnp.ndarray = None,
) -> jnp.ndarray:
    gram = jnp.eye(max(x.shape[0], y.shape[0]))
    return gram[: x.shape[0], : y.shape[0]]


class ReferenceKernelParametersMock(ReferenceKernelParameters):
    pass


class ReferenceKernelMock(ReferenceKernel):
    sigma_matrix = jnp.ones((5, 5))

    def __init__(self, kernel_func: Callable = calculate_reference_gram_mock):
        self.kernel_func = kernel_func
        super().__init__()

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
        full_cov: bool = True,
    ) -> jnp.ndarray:
        gram = self.kernel_func(x, y)
        return gram if full_cov else jnp.diagonal(gram)


class ApproximateKernelParametersMock(ApproximateKernelParameters):
    pass


class ApproximateKernelMock(ApproximateKernel):
    sigma_matrix = jnp.ones((5, 5))

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> ApproximateKernelParametersMock:
        return ApproximateKernelParametersMock()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximateKernelParametersMock:
        return ApproximateKernelParametersMock()

    def _calculate_gram(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        parameters: ApproximateKernelParametersMock = None,
        full_cov: bool = True,
    ) -> jnp.ndarray:
        gram = jnp.ones((x.shape[0], y.shape[0]))
        return gram if full_cov else jnp.diagonal(gram)
