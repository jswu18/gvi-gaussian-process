from abc import ABC
from typing import Any, Dict

from src.parameters.classification_models.classification_models import (
    ClassificationModelParameters,
)
from src.parameters.gaussian_measures.approximate_gaussian_measures import (
    ApproximateGaussianMeasureParameters,
)
from src.parameters.gaussian_measures.reference_gaussian_measures import (
    ReferenceGaussianMeasureParameters,
)


class DistinctMeansClassificationModelParameters(ClassificationModelParameters, ABC):
    pass


class ReferenceDistinctMeansClassificationModelParameters(
    DistinctMeansClassificationModelParameters
):
    gaussian_measures: Dict[Any, ReferenceGaussianMeasureParameters]


class ApproximateDistinctMeansClassificationModelParameters(
    DistinctMeansClassificationModelParameters
):
    gaussian_measures: Dict[Any, ApproximateGaussianMeasureParameters]
