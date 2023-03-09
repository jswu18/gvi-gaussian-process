from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.kernels.kernels import Kernel
from src.mean_functions.mean_functions import MeanFunction
from src.module import Module
from src.parameters.gaussian_measures.gaussian_measure import GaussianMeasureParameters
from src.utils import decorators

PRNGKey = Any  # pylint: disable=invalid-name


class GaussianMeasure(Module, ABC):
    """
    A Gaussian measure defined with respect to a mean function and a kernel.
    """

    Parameters = GaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: Kernel,
    ):
        """
        Defining the training data (x, y), the mean function, and the kernel for the Gaussian measure.
            - n is the number of training points
            - d is the number of dimensions

        Args:
            x: the training inputs design matrix of shape (n, d)
            y: the training outputs response vector of shape (n, 1)
            mean_function: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        self.number_of_train_points = x.shape[0]
        self.x = x
        self.y = y
        self.mean_function = mean_function
        self.kernel = kernel

    @decorators.preprocess_inputs
    @decorators.check_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def calculate_covariance(
        self,
        parameters: GaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance matrix of the Gaussian measure at the sets of points x and y.
        If y is None, the posterior covariance matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance matrix of shape (n, m)

        """
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    @abstractmethod
    def calculate_mean(
        self, parameters: GaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the Gaussian measure at the set of points x.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n, 1)

        """
        raise NotImplementedError

    @abstractmethod
    def compute_expected_log_likelihood(
        self,
        parameters_dict: Union[Dict, FrozenDict],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters_dict: a dictionary containing the parameters, a dictionary is required for jit compilation
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: a scalar representing the empirical expected log likelihood

        """
        raise NotImplementedError

    @staticmethod
    def _compute_expected_log_likelihood(
        mean: jnp.ndarray,
        covariance: jnp.ndarray,
        observation_noise: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        General method for computing the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            mean: response vector from the mean function of shape (n, 1)
            covariance: covariance matrix from the kernel function of shape (n, n)
            observation_noise: the observation noise
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: a scalar representing the empirical expected log likelihood

        """
        return (x.shape[0] / 2) * jnp.log(2 * jnp.pi * observation_noise) + (
            1 / (2 * observation_noise)
        ) * (jnp.sum((y - mean) ** 2) + jnp.trace(covariance))
