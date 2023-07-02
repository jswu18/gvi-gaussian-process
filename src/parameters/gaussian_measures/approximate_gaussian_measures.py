from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.kernels.approximate_kernels import ApproximateKernelParameters
from src.parameters.mean_functions.mean_functions import MeanFunctionParameters


class ApproximateGaussianMeasureParameters(GaussianMeasureParameters):
    mean_function: MeanFunctionParameters
    kernel: ApproximateKernelParameters
