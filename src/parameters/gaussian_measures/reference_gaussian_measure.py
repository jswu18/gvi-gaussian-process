from src.utils.custom_types import JaxFloatType
from src.parameters.gaussian_measures.gaussian_measure import GaussianMeasureParameters
from src.parameters.kernels.kernels import KernelParameters
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class ReferenceGaussianMeasureParameters(GaussianMeasureParameters):
    log_observation_noise: JaxFloatType
    mean_function: MeanFunctionParameters
    kernel: KernelParameters
