from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.kernels.approximation_kernels import ApproximationKernelParameters
from src.parameters.mean_functions.approximation_mean_functions import (
    ApproximationMeanFunctionParameters,
)


class ApproximationGaussianMeasureParameters(GaussianMeasureParameters):
    mean_function: ApproximationMeanFunctionParameters
    kernel: ApproximationKernelParameters
