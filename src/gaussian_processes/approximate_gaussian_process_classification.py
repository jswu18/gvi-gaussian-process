from src.gaussian_processes.base.approximate_base import ApproximateGaussianProcessBase
from src.gaussian_processes.base.classification_base import (
    GaussianProcessClassificationBase,
)


class ApproximateGaussianProcessClassification(
    ApproximateGaussianProcessBase, GaussianProcessClassificationBase
):
    pass
