from abc import ABC

from src.kernels.standard.base import StandardKernelBase, StandardKernelBaseParameters


class NonStationaryKernelBaseParameters(StandardKernelBaseParameters, ABC):
    pass


class NonStationaryKernelBase(StandardKernelBase, ABC):
    pass
