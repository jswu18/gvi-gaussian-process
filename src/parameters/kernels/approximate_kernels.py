from typing import Literal

from src.parameters.kernels.kernels import KernelParameters
from src.utils.custom_types import JaxArrayType


class ApproximateKernelParameters(KernelParameters):
    pass


class StochasticVariationalGaussianProcessKernelParameters(ApproximateKernelParameters):
    """
    log_el_matrix is the logarithm of the L matrix, a lower triangle matrix such that
        sigma_matrix = L @ L.T
    """

    log_el_matrix: JaxArrayType[Literal["float64"]]
