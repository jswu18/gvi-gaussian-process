from src.gps.base.approximate_base import ApproximateGPBase
from src.gps.base.base import GPBaseParameters
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class ApproximateGPRegressionParameters(GPBaseParameters):
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
        super().__init__(mean, kernel)
