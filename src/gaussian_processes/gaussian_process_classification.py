from src.gaussian_processes.base.classification_base import (
    GaussianProcessClassificationBase,
)
from src.gaussian_processes.base.exact_base import ExactGaussianProcessBase


class GaussianProcessClassification(
    ExactGaussianProcessBase, GaussianProcessClassificationBase
):
    pass
