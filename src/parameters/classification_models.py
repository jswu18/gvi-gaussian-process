from abc import ABC
from typing import List

from src.parameters.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasureParameters,
)
from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.gaussian_measures.reference_gaussian_measures import (
    ReferenceGaussianMeasureParameters,
)
from src.parameters.module import ModuleParameters


class ClassificationModelParameters(ModuleParameters, ABC):
    gaussian_measures_parameters: List[GaussianMeasureParameters]


class ReferenceClassificationModelParameters(ClassificationModelParameters):
    gaussian_measures_parameters: List[ReferenceGaussianMeasureParameters]


class ApproximateClassificationModelParameters(ReferenceClassificationModelParameters):
    gaussian_measures_parameters: List[ApproximateGaussianMeasureParameters]
