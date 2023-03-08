from abc import ABC
from typing import Literal

from src.custom_types import ArrayType
from src.parameters.kernels.kernels import KernelParameters


class StandardKernelParameters(KernelParameters, ABC):
    pass


class ARDKernelParameters(StandardKernelParameters):
    log_scaling: float
    log_lengthscales: ArrayType[Literal["float64"]]


class NeuralNetworkGaussianProcessKernelParameters(KernelParameters):
    pass
