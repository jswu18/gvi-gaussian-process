from abc import ABC
from typing import Callable

import jax.numpy as jnp

from src.kernels.base import KernelBase, KernelBaseParameters


class ApproximateBaseKernelParameters(KernelBaseParameters):
    pass


class ApproximateBaseKernel(KernelBase, ABC):
    Parameters = ApproximateBaseKernelParameters

    def __init__(
        self,
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float,
        is_diagonal_regularisation_absolute_scale: bool,
        preprocess_function: Callable = None,
    ):
        self.inducing_points = inducing_points
        self.diagonal_regularisation = diagonal_regularisation
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        super().__init__(
            preprocess_function=preprocess_function,
        )
