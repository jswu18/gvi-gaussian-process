import os

from jax import numpy as jnp

from experiments.shared.data import Data
from experiments.shared.schemas import Actions
from src.inducing_points_selection import InducingPointsSelectorBase
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.custom_types import PRNGKey


def calculate_inducing_points(
    key: PRNGKey,
    inducing_points_selector: InducingPointsSelectorBase,
    data: Data,
    number_of_inducing_points: int,
    kernel: KernelBase,
    kernel_parameters: KernelBaseParameters,
) -> Data:
    x_inducing, inducing_indices = inducing_points_selector.compute_inducing_points(
        key=key,
        training_inputs=jnp.atleast_2d(data.x).reshape(data.x.shape[0], -1),
        number_of_inducing_points=number_of_inducing_points,
        kernel=kernel,
        kernel_parameters=kernel_parameters,
    )
    y_inducing = data.y[inducing_indices]
    return Data(
        x=x_inducing,
        y=y_inducing,
    )


def construct_path(output_path: str, experiment_name: str, action: Actions) -> str:
    return os.path.join(
        output_path,
        action,
        experiment_name,
    )
