from typing import Tuple

from jax import numpy as jnp

from src.inducing_points_selection import ConditionalVarianceInducingPointsSelector
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


def calculate_inducing_points(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    number_of_inducing_points: int,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inducing_points_selector = ConditionalVarianceInducingPointsSelector()
    x_inducing, inducing_indices = inducing_points_selector.compute_inducing_points(
        key=key,
        training_inputs=jnp.atleast_2d(x).reshape(x.shape[0], -1),
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    y_inducing = y[inducing_indices]
    return x_inducing, y_inducing
