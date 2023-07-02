from abc import ABC
from typing import Any, Dict

from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)
from src.parameters.kernels.approximate_kernels import ApproximateKernelParameters
from src.parameters.kernels.kernels import KernelParameters
from src.parameters.mean_functions.mean_functions import (
    NeuralNetworkSharedMeanFunctionParameters,
)
from src.utils.custom_types import JaxFloatType


class SharedMeansClassificationModelParameters(ClassificationModelParameters, ABC):
    pass


class ReferenceSharedMeansClassificationModelParameters(
    SharedMeansClassificationModelParameters
):
    log_observation_noises: Dict[Any, JaxFloatType]
    mean_function: NeuralNetworkSharedMeanFunctionParameters
    kernels: Dict[Any, KernelParameters]


class ApproximateSharedMeansClassificationModelParameters(
    SharedMeansClassificationModelParameters
):
    mean_function: NeuralNetworkSharedMeanFunctionParameters
    kernels: Dict[Any, ApproximateKernelParameters]
