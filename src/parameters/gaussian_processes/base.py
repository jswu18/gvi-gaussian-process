from abc import ABC

from src.parameters.kernels.base import KernelBaseParameters
from src.parameters.mean.base import MeanBaseParameters
from src.parameters.module import ModuleParameters
from src.utils.custom_types import JaxFloatType


class GaussianProcessBaseParameters(ModuleParameters, ABC):
    log_observation_noise: JaxFloatType
    mean: MeanBaseParameters
    kernel: KernelBaseParameters
