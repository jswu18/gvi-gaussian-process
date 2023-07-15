from typing import Literal

from src.parameters.kernels.kernels import KernelParameters
from src.utils.custom_types import JaxArrayType


class ApproximateKernelParameters(KernelParameters):
    pass


class StochasticVariationalGaussianProcessKernelParameters(ApproximateKernelParameters):
    """
    el_matrix_lower_triangle is a lower triangle of the L matrix
    el_matrix_log_diagonal is the logarithm of the diagonal of the L matrix
    combining them such that:
        L = el_matrix_lower_triangle + diagonalise(exp(el_matrix_log_diagonal))
    and
        sigma_matrix = L @ L.T
    """

    el_matrix_lower_triangle: JaxArrayType[Literal["float64"]]
    el_matrix_log_diagonal: JaxArrayType[Literal["float64"]]
