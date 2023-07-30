from abc import ABC
from typing import Callable

import jax.numpy as jnp

from src.kernels.base import KernelBase, KernelBaseParameters


class ApproximateBaseKernelParameters(KernelBaseParameters):
    pass


class ApproximateBaseKernel(KernelBase, ABC):
    """
    Approximate kernels which are defined with respect to a reference kernel
    """

    Parameters = ApproximateBaseKernelParameters

    def __init__(
        self,
        reference_kernel_parameters: KernelBaseParameters,
        reference_kernel: KernelBase,
        preprocess_function: Callable[[jnp.ndarray], jnp.ndarray] = None,
    ):
        """
        Defining the kernel with respect to a reference Gaussian measure.

        Args:
            reference_kernel_parameters: the parameters of the reference kernel.
            reference_kernel: the kernel of the reference Gaussian measure.
        """
        self.reference_kernel = reference_kernel
        self.reference_kernel_parameters = reference_kernel_parameters
        super().__init__(preprocess_function=preprocess_function)
