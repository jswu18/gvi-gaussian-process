from abc import ABC
from typing import Callable

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor

from src.kernels.approximate.base import (
    ApproximateBaseKernel,
    ApproximateBaseKernelParameters,
)
from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.matrix_operations import add_diagonal_regulariser


class ExtendedSVGPBaseKernelParameters(ApproximateBaseKernelParameters, ABC):
    pass


class ExtendedSVGPBaseKernel(ApproximateBaseKernel, ABC):
    """
    Approximate kernels which are defined with respect to a reference kernel
    """

    Parameters = ExtendedSVGPBaseKernelParameters

    def __init__(
        self,
        reference_kernel: KernelBase,
        reference_kernel_parameters: KernelBaseParameters,
        log_observation_noise: float,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Defining the stochastic variational Gaussian process kernel using the reference Gaussian measure
        and inducing points.

        Args:
            reference_kernel_parameters: the parameters of the reference kernel.
            log_observation_noise: the log observation noise of the model
            reference_kernel: the kernel of the reference Gaussian measure.
            inducing_points: the inducing points of the stochastic variational Gaussian process.
            training_points: the training points of the stochastic variational Gaussian process.
            is_diagonal_regularisation_absolute_scale: whether the diagonal regularisation is an absolute scale.
        """
        self.reference_kernel = reference_kernel
        self.reference_kernel_parameters = reference_kernel_parameters
        self.log_observation_noise = log_observation_noise
        self.number_of_dimensions = inducing_points.shape[1]
        self.training_points = training_points
        self.reference_gram_inducing = self.reference_kernel.calculate_gram(
            parameters=reference_kernel_parameters,
            x1=inducing_points,
            x2=inducing_points,
        )
        self.reference_gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=self.reference_gram_inducing,
                diagonal_regularisation=diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            )
        )
        self.gram_inducing_train = self.reference_kernel.calculate_gram(
            parameters=reference_kernel_parameters,
            x1=inducing_points,
            x2=training_points,
        )
        ApproximateBaseKernel.__init__(
            self,
            inducing_points=inducing_points,
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            preprocess_function=preprocess_function,
        )
