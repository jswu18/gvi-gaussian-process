from typing import Dict, List, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.base import KernelBase, KernelBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG


class MultiOutputKernelParameters(KernelBaseParameters):
    kernels: List[KernelBaseParameters]


class MultiOutputKernel(KernelBase):
    Parameters = MultiOutputKernelParameters

    def __init__(self, kernels: List[KernelBase]):
        assert all(kernel.number_output_dimensions == 1 for kernel in kernels)
        self.kernels = kernels
        super().__init__(
            number_output_dimensions=len(kernels),
            preprocess_function=None,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> MultiOutputKernelParameters:
        assert len(parameters["kernels"]) == self.number_output_dimensions
        return MultiOutputKernel.Parameters(
            kernels=[
                kernel.generate_parameters(parameters=parameters_)
                for kernel, parameters_ in zip(self.kernels, parameters["kernels"])
            ]
        )

    def _calculate_gram(
        self,
        parameters: Union[Dict, FrozenDict, MultiOutputKernelParameters],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the prior gram matrix of multiple kernels.
            - k is the number of kernels
            - m1 is the number of points in x1
            - m2 is the number of points in x2
            - d is the number of dimensions

        Args:
            parameters: parameters of the kernel
            x1: design matrix of shape (m1, d)
            x2: design matrix of shape (m2, d)

        Returns: the kernel a stacked gram matrix of shape (k, m_1, m_2)
        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return jnp.array(
            [
                kernel_.calculate_gram(
                    parameters=parameters_,
                    x1=x1,
                    x2=x2,
                )
                for kernel_, parameters_ in zip(self.kernels, parameters.kernels)
            ]
        )
