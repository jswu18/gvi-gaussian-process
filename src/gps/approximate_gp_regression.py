from typing import Dict, Union

import pydantic
from flax.core import FrozenDict

from src.gps.base.approximate_base import ApproximateGPBase, ApproximateGPBaseParameters
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase
from src.utils.custom_types import PRNGKey


class ApproximateGPRegressionParameters(ApproximateGPBaseParameters):
    pass


class ApproximateGPRegression(ApproximateGPBase, GPRegressionBase):

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

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGPRegressionParameters:
        """
        Generates a Pydantic model of the parameters for Regulariser Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Regulariser Gaussian Measures.

        Returns: A Pydantic model of the parameters for Regulariser Gaussian Measures.

        """
        return ApproximateGPRegression.Parameters(
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )
