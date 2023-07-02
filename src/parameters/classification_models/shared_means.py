from typing import Any, Dict

from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)
from src.parameters.kernels.kernels import KernelParameters
from src.parameters.mean_functions.classification_shared_mean_functions import (
    SharedMeanFunctionParameters,
)


class SharedMeansClassificationModelParameters(ClassificationModelParameters):
    mean_function: SharedMeanFunctionParameters
    kernels: Dict[Any, KernelParameters]
