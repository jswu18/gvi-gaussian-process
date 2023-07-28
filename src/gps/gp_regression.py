from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from src.gps.base.base import GPBaseParameters
from src.gps.base.exact_base import ExactGPBase
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.utils.custom_types import PRNGKey


class GPRegressionParameters(GPBaseParameters):
    pass


class GPRegression(ExactGPBase, GPRegressionBase):

    Parameters = GPRegressionParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian process.

        Args:
            mean: the mean function of the Gaussian process
            kernel: the kernel of the Gaussian process
        """
        GPRegressionBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
        )
        ExactGPBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
            x=x,
            y=y,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> GPRegressionParameters:
        """
        Generates a Pydantic model of the parameters for Reference Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Reference Gaussian Measures.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        return GPRegression.Parameters(
            log_observation_noise=parameters["log_observation_noise"],
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> GPRegressionParameters:
        """
        Initialise each parameter of the Reference Gaussian measure with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Reference Gaussian Measures.

        """
        return GPRegression.Parameters(
            log_observation_noise=jax.random.normal(key),
            mean=self.mean.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )
