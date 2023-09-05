from typing import Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from src.gps.base.base import GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.gps.base.exact_base import ExactGPBase
from src.kernels import TemperedKernel, TemperedKernelParameters
from src.kernels.multi_output_kernel import (
    MultiOutputKernel,
    MultiOutputKernelParameters,
)
from src.means.base import MeanBase
from src.module import PYDANTIC_VALIDATION_CONFIG


class GPClassificationParameters(GPBaseParameters):
    """
    The parameters of an exact Gaussian process classification model.
    The parameters are the mean function, the kernel, and the observation noise.
    The kernel is either a multi-output kernel or a tempered kernel (for multi-class classification).
    """

    kernel: Union[MultiOutputKernelParameters, TemperedKernelParameters]


class GPClassification(ExactGPBase, GPClassificationBase):
    """
    A Gaussian process classification model.
    """

    Parameters = GPClassificationParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: Union[MultiOutputKernel, TemperedKernel],
        x: jnp.ndarray,
        y: jnp.ndarray,
        epsilon: float = 0.01,
        hermite_polynomial_order: int = 50,
        cdf_lower_bound: float = 1e-10,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian process.

        Args:
            mean: the mean function of the Gaussian process
            kernel: the kernel of the Gaussian process
        """
        ExactGPBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
            x=x,
            y=y,
        )
        GPClassificationBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
            cdf_lower_bound=cdf_lower_bound,
        )
        self.mean = mean
        self.kernel = kernel

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> GPClassificationParameters:
        """
        Generates a Pydantic model of the parameters for exact Gaussian process classification.

        Args:
            parameters: A dictionary of the parameters for exact Gaussian process classification.

        Returns: A Pydantic model of the parameters for exact Gaussian process classification.

        """
        return GPClassification.Parameters(
            log_observation_noise=parameters["log_observation_noise"],
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )
