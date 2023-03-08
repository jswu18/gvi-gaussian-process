from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.kernels.kernels import KernelParameters
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class ReferenceGaussianMeasureParameters(GaussianMeasureParameters):
    log_observation_noise: float
    mean_function: MeanFunctionParameters
    kernel: KernelParameters
