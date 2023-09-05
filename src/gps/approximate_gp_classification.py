from typing import Dict, Union

import pydantic
from flax.core import FrozenDict

from src.gps.base.approximate_base import ApproximateGPBase, ApproximateGPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.kernels import TemperedKernel
from src.kernels.multi_output_kernel import MultiOutputKernel
from src.means.base import MeanBase
from src.utils.custom_types import PRNGKey


class ApproximateGPClassificationParameters(ApproximateGPBaseParameters):
    pass


class ApproximateGPClassification(ApproximateGPBase, GPClassificationBase):

    Parameters = ApproximateGPClassificationParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: Union[MultiOutputKernel, TemperedKernel],
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
        ApproximateGPBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
        )
        GPClassificationBase.__init__(
            self,
            mean=mean,
            kernel=kernel,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
            cdf_lower_bound=cdf_lower_bound,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGPClassificationParameters:
        """
        Generates a Pydantic model of the parameters for Regulariser Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Regulariser Gaussian Measures.

        Returns: A Pydantic model of the parameters for Regulariser Gaussian Measures.

        """
        return ApproximateGPClassification.Parameters(
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )
