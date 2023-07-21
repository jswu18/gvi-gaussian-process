from src.gps.base.approximate_base import ApproximateGPBase
from src.gps.base.base import GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class ApproximateGPClassificationParameters(GPBaseParameters):
    pass


class ApproximateGPClassification(ApproximateGPBase, GPClassificationBase):

    Parameters = ApproximateGPClassificationParameters

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
        epsilon: float,
        hermite_polynomial_order: int,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian process.

        Args:
            mean: the mean function of the Gaussian process
            kernel: the kernel of the Gaussian process
        """
        super().__init__(
            mean=mean,
            kernel=kernel,
            epsilon=epsilon,
            hermite_polynomial_order=hermite_polynomial_order,
        )
