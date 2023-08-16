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
        preprocess_function: Callable = None,
    ):
        self.inducing_points = inducing_points
        super().__init__(
            preprocess_function=preprocess_function,
        )
