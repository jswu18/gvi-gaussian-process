from abc import ABC
from typing import Literal

from src.parameters.kernels.kernels import KernelParameters
from src.utils.custom_types import JaxArrayType, JaxFloatType


class StandardKernelParameters(KernelParameters, ABC):
    pass


class ARDKernelParameters(StandardKernelParameters):
    log_scaling: JaxFloatType
    log_lengthscales: JaxArrayType[Literal["float64"]]


class NeuralNetworkGaussianProcessKernelParameters(KernelParameters):
    pass
