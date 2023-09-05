from abc import ABC

from src.kernels.standard.base import StandardKernelBase, StandardKernelBaseParameters


class NonStationaryKernelBaseParameters(StandardKernelBaseParameters, ABC):
    """
    Parameters for non-stationary kernels.
    """

    pass


class NonStationaryKernelBase(StandardKernelBase, ABC):
    """
    Non-stationary kernels are kernels that are not stationary. This means that
    the kernel function depends on the inputs x and y and not only on their
    difference x - y. This class is a base class for all non-stationary kernels.
    """

    pass
