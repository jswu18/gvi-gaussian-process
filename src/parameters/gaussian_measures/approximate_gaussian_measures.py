from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.kernels.approximate_kernels import ApproximateKernelParameters
from src.parameters.mean_functions.approximate_mean_functions import (
    ApproximateMeanFunctionParameters,
)


class ApproximateGaussianMeasureParameters(GaussianMeasureParameters):
    mean_function: ApproximateMeanFunctionParameters
    kernel: ApproximateKernelParameters
