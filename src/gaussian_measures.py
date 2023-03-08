from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import random
from jax.scipy.linalg import cho_factor, cho_solve

from src import decorators
from src.kernels.approximation_kernels import ApproximationKernel
from src.kernels.kernels import Kernel
from src.kernels.reference_kernels import StandardKernel
from src.mean_functions.approximation_mean_functions import ApproximationMeanFunction
from src.mean_functions.mean_functions import MeanFunction
from src.module import Module
from src.parameters.gaussian_measures.approximation_gaussian_measures import (
    ApproximationGaussianMeasureParameters,
)
from src.parameters.gaussian_measures.gaussian_measures import GaussianMeasureParameters
from src.parameters.gaussian_measures.reference_gaussian_measures import (
    ReferenceGaussianMeasureParameters,
)

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
        Defining the training data, the mean function and the kernel for the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: the training inputs of shape (n, d)
            y: the training outputs of shape (n, 1)
            mean_function: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        self.number_of_train_points = x.shape[0]
        self.x = x
        self.y = y
        self.mean_function = mean_function
        self.kernel = kernel

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
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
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)
            parameters: parameters of the Gaussian measure

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
            x: design matrix of shape (n, d)
            parameters: parameters of the Gaussian measure

        Returns: the mean function evaluations of shape (n, 1)

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


        Returns: a scalar representing the empirical expected log likelihood

        """
        raise NotImplementedError

    def _compute_expected_log_likelihood(
        self,
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
        Returns: a scalar representing the empirical expected log likelihood

        """
        return (x.shape[0] / 2) * jnp.log(2 * jnp.pi * observation_noise) + (
            1 / (2 * observation_noise)
        ) * (jnp.sum((y - mean) ** 2) + jnp.trace(covariance))


class ReferenceGaussianMeasure(GaussianMeasure):
    Parameters = ReferenceGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: MeanFunction,
        kernel: StandardKernel,
    ):
        """
        Defining the training data, the mean function and the kernel for the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: the training inputs of shape (n, d)
            y: the training outputs of shape (n, 1)
            mean_function: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        super().__init__(x, y, mean_function, kernel)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ReferenceGaussianMeasureParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return ReferenceGaussianMeasureParameters(
            log_observation_noise=parameters["log_observation_noise"],
            mean_function=self.mean_function.generate_parameters(
                parameters["mean_function"]
            ),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ReferenceGaussianMeasureParameters:
        """
        Initialise each parameter of the Gaussian measure with the appropriate random initialisation.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        return ReferenceGaussianMeasureParameters(
            log_observation_noise=random.normal(key),
            mean_function=self.mean_function.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )

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
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)
            parameters_dict: parameters of the Gaussian measure

        Returns: a scalar representing the empirical expected log likelihood

        """
        parameters = self.generate_parameters(parameters_dict)
        return self._compute_expected_log_likelihood(
            mean=self.calculate_mean(x=x, parameters=parameters),
            covariance=self.calculate_covariance(x=x, parameters=parameters),
            observation_noise=jnp.exp(parameters.log_observation_noise),
            x=x,
            y=y,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_mean(
        self, parameters: ReferenceGaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean using the formula for the posterior mean of a Gaussian process which is
        m(x) = k(x, X) @ (K(X, X) + σ^2I)^-1 @ y. This is added to the mean function prediction to generate the
        full posterior mean.

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "log_observation_noise": float,
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            x: design matrix of shape (n, d)
            parameters: parameters of the Gaussian measure

        Returns: the mean function evaluations of shape (n, 1)

        """
        gram_train = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x=self.x,
        )
        gram_train_test = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x=self.x,
            y=x,
        )
        observation_noise = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = cho_factor(gram_train + observation_noise)
        kernel_mean = gram_train_test.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=self.y
        )
        return kernel_mean + self.mean_function.predict(
            x=x, parameters=parameters.mean_function
        )

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_covariance(
        self,
        parameters: ReferenceGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance using the formula for the posterior covariance of a Gaussian process which is
        k(x, y) = k(x, y) - k(x, X) @ (K(X, X) + σ^2I)^-1 @ k(X, y).
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "log_observation_noise": float,
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance of shape (n, m)

        """
        gram_train = self.kernel.calculate_gram(x=self.x, parameters=parameters.kernel)
        gram_train_x = self.kernel.calculate_gram(
            x=self.x, y=x, parameters=parameters.kernel
        )
        gram_train_y = self.kernel.calculate_gram(
            x=self.x, y=y, parameters=parameters.kernel
        )
        gram_xy = self.kernel.calculate_gram(x=x, y=y, parameters=parameters.kernel)
        observation_noise_matrix = jnp.eye(self.number_of_train_points) * jnp.exp(
            parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = cho_factor(
            gram_train + observation_noise_matrix
        )

        return gram_xy - gram_train_x.T @ cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_y
        )


class ApproximationGaussianMeasure(GaussianMeasure):
    """
    A Gaussian measure which uses an approximation to the kernel and mean function. These are defined with respect
    to reference Gaussian measures.
    """

    Parameters = ApproximationGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        mean_function: ApproximationMeanFunction,
        kernel: ApproximationKernel,
    ):
        """
        Defining the training data, the mean function and the kernel for the Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: the training inputs of shape (n, d)
            y: the training outputs of shape (n, 1)
            mean_function: the approximation mean function of the Gaussian measure
            kernel: the approximation kernel of the Gaussian measure
        """
        super().__init__(x, y, mean_function, kernel)
        self.kernel = kernel

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> ApproximationGaussianMeasureParameters:
        """
        Generator for a Pydantic model of the parameters for the module.
        Args:
            parameters: A dictionary of the parameters of the module.

        Returns: A Pydantic model of the parameters for the module.

        """
        return ApproximationGaussianMeasureParameters(
            reference_gaussian_measure=self.kernel.reference_gaussian_measure_parameters,
            mean_function=self.mean_function.generate_parameters(
                parameters["mean_function"]
            ),
            kernel=self.kernel.generate_parameters(parameters["kernel"]),
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> ApproximationGaussianMeasureParameters:
        """
        Initialise each parameter of the Gaussian measure with the appropriate random initialisation.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        return ApproximationGaussianMeasureParameters(
            mean_function=self.mean_function.initialise_random_parameters(key),
            kernel=self.kernel.initialise_random_parameters(key),
        )

    def compute_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, ApproximationGaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)
            parameters: parameters of the Gaussian measure

        Returns: a scalar representing the empirical expected log likelihood

        """
        if not isinstance(parameters, ApproximationGaussianMeasureParameters):
            parameters = self.generate_parameters(parameters)
        return self._compute_expected_log_likelihood(
            mean=self.calculate_mean(x=x, parameters=parameters),
            covariance=self.calculate_covariance(x=x, parameters=parameters),
            observation_noise=jnp.exp(
                self.kernel.reference_gaussian_measure_parameters.log_observation_noise
            ),
            x=x,
            y=y,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_mean(
        self, parameters: ApproximationGaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean by evaluating the mean function at the test points.
            - n is the number of points in x
            - d is the number of dimensions

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            x: design matrix of shape (n, d)
            parameters: parameters of the Gaussian measure

        Returns: the mean function evaluations of shape (n, 1)

        """
        return self.mean_function.predict(x=x, parameters=parameters.mean_function)

    @decorators.common.default_duplicate_x
    @decorators.kernels.preprocess_kernel_inputs
    @decorators.kernels.check_kernel_inputs
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_covariance(
        self,
        parameters: ApproximationGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance by evaluating the kernel gram matrix at the test points.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        The parameters of the Gaussian measure are passed in as a dictionary of the form:
            parameters = {
                "mean_function": FrozenDict, the parameters of the mean function,
                "kernel": FrozenDict, the parameters of the kernel,
            }

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance of shape (n, m)

        """
        return self.kernel.calculate_gram(x=x, y=y, parameters=parameters.kernel)
