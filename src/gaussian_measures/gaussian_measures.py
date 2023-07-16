from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict
from jax import jit, vmap

from src.kernels.kernels import Kernel
from src.mean_functions.mean_functions import MeanFunction
from src.module import Module
from src.parameters.gaussian_measures.gaussian_measures import (
    GaussianMeasureParameters,
    TemperedGaussianMeasureParameters,
)
from src.utils.custom_types import JaxFloatType

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

        # define a jit-compiled function to compute the negative expected log likelihood
        self._is_jit_negative_expected_log_likelihood_warmed_up = False
        self._jit_compiled_compute_negative_expected_log_likelihood = jit(
            lambda parameters_dict, x, y: (
                self._compute_negative_expected_log_likelihood(
                    x=x,
                    y=y,
                    parameters=parameters_dict,
                )
            )
        )

    @abstractmethod
    def _calculate_covariance(
        self,
        parameters: GaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance matrix of the Gaussian measure at the sets of points x and y.
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

    @abstractmethod
    def _calculate_mean(
        self, parameters: GaussianMeasureParameters, x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the Gaussian measure at the set of points x.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n,)

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_observation_noise(
        self, parameters: GaussianMeasureParameters
    ) -> JaxFloatType:
        """
        Extracts the observation noise of the Gaussian measure from the parameters.

        Args:
            parameters: parameters of the Gaussian measure

        Returns: the observation noise

        """
        raise NotImplementedError

    @abstractmethod
    def _compute_negative_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, GaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: a scalar representing the empirical expected log likelihood

        """
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_observation_noise(
        self, parameters: GaussianMeasureParameters
    ) -> JaxFloatType:
        """
        Extracts the observation noise of the Gaussian measure from the parameters
        Calls _calculate_observation_noise, which needs to be implemented by the child class.

        Args:
            parameters: parameters of the Gaussian measure

        Returns: the observation noise

        """
        Module.check_parameters(parameters, self.Parameters)
        return self._calculate_observation_noise(parameters=parameters)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        full_cov: bool = True,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance matrix of the Gaussian measure at the sets of points x and y.
        Calls _calculate_covariance, which needs to be implemented by the child class.
        If y is None, the posterior covariance matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)
            full_cov: Whether to calculate full covariance matrix or just the diagonal

        Returns: the posterior covariance matrix of shape (n, m)

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        x, y = self.kernel.preprocess_inputs(x, y)
        self.kernel.check_inputs(x, y)
        if full_cov:
            return self._calculate_covariance(parameters=parameters, x=x, y=y)
        else:
            return vmap(
                lambda x_, y_: self._calculate_covariance(
                    parameters=parameters,
                    x=x_[None, ...],
                    y=y_[None, ...],
                )
            )(x, y).reshape(
                -1,
            )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_mean(
        self,
        parameters: Union[Dict, FrozenDict, GaussianMeasureParameters],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the Gaussian measure at the set of points x.
        Calls _calculate_mean, which needs to be implemented by the child class.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n,)

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        x = self.mean_function.preprocess_input(x)
        return self._calculate_mean(parameters=parameters, x=x)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def compute_negative_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, GaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Jit needs a dictionary of parameters to be passed to it to allow for jit compilation.

            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: the negative expected log likelihood of the gaussian measure

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        if not self._is_jit_negative_expected_log_likelihood_warmed_up:
            self._jit_compiled_compute_negative_expected_log_likelihood(
                parameters.dict(), x[:2, ...], y[:2, ...]
            )
            self._is_jit_negative_expected_log_likelihood_warmed_up = True
        return self._jit_compiled_compute_negative_expected_log_likelihood(
            parameters_dict=parameters.dict(),
            x=x,
            y=y,
        )


class TemperedGaussianMeasure(GaussianMeasure):
    """
    Provides a tempered version of the Gaussian measure where the covariance matrix is multiplied by a tempering factor.
    The tempered Gaussian measure is defined with respect to an existing Gaussian measure.
    """

    Parameters = TemperedGaussianMeasureParameters

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        gaussian_measure: GaussianMeasure,
        gaussian_measure_parameters: GaussianMeasureParameters,
    ):
        """
        Defining the training data (x, y), and the untempered Gaussian measure.
            - n is the number of training points
            - d is the number of dimensions

        Args:
            x: the training inputs design matrix of shape (n, d)
            y: the training outputs response vector of shape (n, 1)
            gaussian_measure: the untempered Gaussian measure
            gaussian_measure_parameters: the parameters of the untempered Gaussian measure
        """
        super().__init__(
            x=x,
            y=y,
            mean_function=gaussian_measure.mean_function,
            kernel=gaussian_measure.kernel,
        )
        self.gaussian_measure = gaussian_measure
        self.gaussian_measure_parameters = gaussian_measure_parameters

    def _calculate_covariance(
        self,
        parameters: TemperedGaussianMeasureParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate the posterior covariance matrix of the tempered Gaussian measure at the set of points x and y
        by scaling the untempered Gaussian measure covariance by the tempering factor.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the posterior covariance matrix of shape (n, m)

        """
        return jnp.exp(
            parameters.log_tempering_factor
        ) * self.gaussian_measure.calculate_covariance(
            parameters=self.gaussian_measure_parameters,
            x=x,
            y=y,
        )

    def _calculate_mean(
        self, x: jnp.ndarray, parameters: TemperedGaussianMeasureParameters = None
    ) -> jnp.ndarray:
        """
        Calculate the posterior mean of the Gaussian measure at the set of points x by calling the mean function
        of the untempered Gaussian measure.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)

        Returns: the mean function evaluations, a vector of shape (n,)

        """
        return self.gaussian_measure.calculate_mean(
            parameters=self.gaussian_measure_parameters, x=x
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[Dict, FrozenDict]
    ) -> TemperedGaussianMeasureParameters:
        """
        Generates a Pydantic model of the parameters for Tempered Gaussian Measures.

        Args:
            parameters: A dictionary of the parameters for Tempered Gaussian Measures.

        Returns: A Pydantic model of the parameters for Tempered Gaussian Measures.

        """
        return TemperedGaussianMeasure.Parameters(
            log_tempering_factor=parameters["log_tempering_factor"],
        )

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> TemperedGaussianMeasureParameters:
        """
        Initialise each parameter of the Tempered Gaussian measure with the appropriate random initialisation.

        Args:
            key: A random key used to initialise the parameters.

        Returns: A Pydantic model of the parameters for Tempered Gaussian Measures.

        """
        pass

    def _calculate_observation_noise(
        self, parameters: TemperedGaussianMeasureParameters = None
    ) -> JaxFloatType:
        """
        Extracts the observation noise of the untempered Gaussian measure its parameters.

        Args:
            parameters: parameters of the tempered Gaussian measure (unused)

        Returns: the observation noise

        """
        return self.gaussian_measure.calculate_observation_noise(
            parameters=self.gaussian_measure_parameters
        )

    def _compute_negative_expected_log_likelihood(
        self,
        parameters: Union[Dict, FrozenDict, TemperedGaussianMeasureParameters],
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> float:
        """
        Compute the expected log likelihood of the Gaussian measure at the inputs x and outputs y.
            - n is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: a dictionary or Pydantic model containing the parameters,
                        a dictionary is required for jit compilation which is converted if necessary
            x: design matrix of shape (n, d)
            y: response vector of shape (n, 1)

        Returns: a scalar representing the empirical expected log likelihood

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        mean = self.calculate_mean(x=x, parameters=parameters)
        covariance = self.calculate_covariance(x=x, parameters=parameters)
        observation_noise = self.calculate_observation_noise(parameters=parameters)

        diagonal_covariance = jnp.exp(parameters.log_tempering_factor) * (
            jnp.diag(covariance) + observation_noise
        )
        error = y - mean

        return (x.shape[0] / 2) * (
            jnp.log(2 * jnp.pi)
            + jnp.sum(jnp.log(diagonal_covariance))
            + error.T @ jnp.diag(1 / diagonal_covariance) @ error
        )
