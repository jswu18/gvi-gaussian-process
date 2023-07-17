from src.gaussian_processes.base.exact_base import ExactGaussianProcessBase
from src.gaussian_processes.base.regression_base import GaussianProcessRegressionBase


class GaussianProcessRegression(
    ExactGaussianProcessBase, GaussianProcessRegressionBase
):
    pass
