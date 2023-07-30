from typing import Any, Callable, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


def calculate_reference_gram_mock(
    x1: jnp.ndarray,
    x2: jnp.ndarray = None,
) -> jnp.ndarray:
    return jnp.ones((x1.shape[0], x2.shape[0]))


def calculate_reference_gram_eye_mock(
    x1: jnp.ndarray,
    x2: jnp.ndarray = None,
) -> jnp.ndarray:
    gram = jnp.eye(max(x1.shape[0], x2.shape[0]))
    return gram[: x1.shape[0], : x2.shape[0]]


class MockKernelParameters(KernelBaseParameters):
    pass


class MockKernel(KernelBase):
    sigma_matrix = jnp.ones((5, 5))

    def __init__(self, kernel_func: Callable = calculate_reference_gram_mock):
        self.kernel_func = kernel_func
        super().__init__()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockKernelParameters:
        return MockKernelParameters()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MockKernelParameters:
        return MockKernelParameters()

    def _calculate_gram(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray = None,
        parameters: MockKernelParameters = None,
        full_cov: bool = True,
    ) -> jnp.ndarray:
        gram = self.kernel_func(x1, x2)
        return gram if full_cov else jnp.diagonal(gram)


class MockApproximateKernelParameters(ApproximateBaseKernelParameters):
    pass


class MockApproximateKernel(ApproximateBaseKernel):
    sigma_matrix = jnp.ones((5, 5))

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> MockApproximateKernelParameters:
        return MockApproximateKernelParameters()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> MockApproximateKernelParameters:
        return MockApproximateKernelParameters()

    @staticmethod
    def _calculate_gram(
        x1: jnp.ndarray,
        x2: jnp.ndarray = None,
        parameters: MockApproximateKernelParameters = None,
        full_cov: bool = True,
    ) -> jnp.ndarray:
        gram = jnp.ones((x1.shape[0], x2.shape[0]))
        return gram if full_cov else jnp.diagonal(gram)
