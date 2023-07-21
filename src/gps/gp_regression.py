from src.gps.base.base import GPBaseParameters
from src.gps.base.exact_base import ExactGPBase
from src.gps.base.regression_base import GPRegressionBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class GPRegressionParameters(GPBaseParameters):
    pass


class GPRegression(ExactGPBase, GPRegressionBase):

    Parameters = GPRegressionParameters

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
