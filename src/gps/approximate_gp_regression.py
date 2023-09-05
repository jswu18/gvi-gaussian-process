from typing import Dict, Union

import pydantic
from flax.core import FrozenDict

from src.gps.base.approximate_base import ApproximateGPBase, ApproximateGPBaseParameters
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.module import PYDANTIC_VALIDATION_CONFIG


class ApproximateGPRegressionParameters(ApproximateGPBaseParameters):
    """
    A Pydantic model of the parameters for an approximate Gaussian process regressor.
    This will contain the parameters for the mean function, and the kernel of the Gaussian process.
    """

    pass


class ApproximateGPRegression(ApproximateGPBase, GPRegressionBase):
    """
    An approximate Gaussian process regressor.
    """

    Parameters = ApproximateGPRegressionParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian process.

        Args:
            mean: the mean function of the Gaussian process
            kernel: the kernel of the Gaussian process
        """
        ApproximateGPBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
        )
        GPRegressionBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
        )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGPRegressionParameters:
        """
        Generates a Pydantic model of the parameters for an approximate Gaussian process regressor.

        Args:
            parameters: A dictionary of the parameters for an approximate Gaussian process regressor.

        Returns: A Pydantic model of the parameters for an approximate Gaussian process regressor.

        """
        return ApproximateGPRegression.Parameters(
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )
