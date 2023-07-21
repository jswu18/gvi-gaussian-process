from src.gps.base.base import GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.gps.base.exact_base import ExactGPBase
from src.kernels.base import KernelBase
from src.means.base import MeanBase


class GPClassificationParameters(GPBaseParameters):
    pass


class GPClassification(ExactGPBase, GPClassificationBase):

    Parameters = GPClassificationParameters

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
