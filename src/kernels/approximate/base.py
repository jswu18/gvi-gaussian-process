from abc import ABC
from typing import Callable

import jax.numpy as jnp

from src.kernels.base import KernelBase, KernelBaseParameters


class ApproximateBaseKernelParameters(KernelBaseParameters):
    pass


class ApproximateBaseKernel(KernelBase, ABC):
    """
    Base class for approximate kernels.
    All approximate kernels will inherit from this class.
    """

    Parameters = ApproximateBaseKernelParameters

    def __init__(
        self,
        inducing_points: jnp.ndarray,
        diagonal_regularisation: float,
        is_diagonal_regularisation_absolute_scale: bool,
        preprocess_function: Callable = None,
    ):
        """
        A constructor for approximate kernels.

        Args:
            inducing_points: all approximate kernels are constructed with inducing points
            diagonal_regularisation: the diagonal regularisation of the kernel during the Cholesky decomposition.
            is_diagonal_regularisation_absolute_scale: whether the diagonal regularisation is an absolute scale.
            preprocess_function: a function to preprocess the data before calculating the kernel.
        """
        self.inducing_points = inducing_points
        self.diagonal_regularisation = diagonal_regularisation
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        super().__init__(
            preprocess_function=preprocess_function,
        )
