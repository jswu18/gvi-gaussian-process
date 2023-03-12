from abc import ABC
from typing import Literal

from src.parameters.custom_types import ArrayType, JaxFloatType
from src.parameters.kernels.kernels import KernelParameters


class StandardKernelParameters(KernelParameters, ABC):
    pass


class ARDKernelParameters(StandardKernelParameters):
    log_scaling: JaxFloatType
    log_lengthscales: ArrayType[Literal["float64"]]


class NeuralNetworkGaussianProcessKernelParameters(KernelParameters):
    pass
