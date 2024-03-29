from typing import Dict, Union

import pydantic
from flax.core import FrozenDict

from src.gps.base.approximate_base import ApproximateGPBase, ApproximateGPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.kernels import TemperedKernel
from src.kernels.multi_output_kernel import MultiOutputKernel
from src.means.base import MeanBase
from src.module import PYDANTIC_VALIDATION_CONFIG


class ApproximateGPClassificationParameters(ApproximateGPBaseParameters):
    """
    A Pydantic model of the parameters for an approximate Gaussian process classifier.
    This will contain the parameters for the mean function, and the kernel of the Gaussian process.
    """

    pass


class ApproximateGPClassification(ApproximateGPBase, GPClassificationBase):
    """
    An approximate Gaussian process classifier.
    """

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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximateGPClassificationParameters:
        """
        Generates a Pydantic model of the parameters for an approximate Gaussian process classifier.

        Args:
            parameters: A dictionary of the parameters for an approximate Gaussian process classifier.

        Returns: A Pydantic model of the parameters for an approximate Gaussian process classifier.

        """
        return ApproximateGPClassification.Parameters(
            mean=self.mean.generate_parameters(parameters["mean"]),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )
