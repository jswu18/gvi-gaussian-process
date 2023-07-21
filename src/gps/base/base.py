from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.distributions import Distribution, Gaussian
from src.kernels.base import KernelBase, KernelBaseParameters
from src.means.base import MeanBase, MeanBaseParameters
from src.module import Module, ModuleParameters
from src.utils.custom_types import JaxFloatType
from src.utils.jit_compiler import JitCompiler

PRNGKey = Any  # pylint: disable=invalid-name


class GPBaseParameters(ModuleParameters, ABC):
    log_observation_noise: JaxFloatType
    mean: MeanBaseParameters
    kernel: KernelBaseParameters


class GPBase(Module, ABC):
    """
    A Gaussian measure defined with respect to a mean function and a kernel.
    """

    Parameters = GPBaseParameters
    PredictDistribution = Distribution

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian measure.

        Args:
            mean: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        self.mean = mean
        self.kernel = kernel
        self._jit_compiled_predict_probability = JitCompiler(self._predict_probability)
        super().__init__(preprocess_function=None)

    @abstractmethod
    def _calculate_prediction_distribution(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Gaussian:
        raise NotImplementedError

    @abstractmethod
    def _predict_probability(
        self, parameters: GPBaseParameters, x: jnp.ndarray
    ) -> Distribution:
        raise NotImplementedError

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_probability(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
    ) -> Distribution:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        probabilities = self._jit_compiled_predict_probability(parameters.dict(), x)
        return self.PredictDistribution(*probabilities)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prior_distribution(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> Gaussian:
        """
        Calculate the prior distribution of the Gaussian Processes.
            - m is the number of points in x
            - d is the number of dimensions

        Args:
            parameters: parameters of the Gaussian measure
            x: design matrix of shape (n, d)
            full_covariance: whether to compute the full covariance matrix or just the diagonal (requires m1 == m2)

        Returns: the prior distribution

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        return Gaussian(
            mean=self.mean.predict(parameters.mean, x),
            covariance=self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x,
                x2=x,
                full_covariance=full_covariance,
            ),
            full_covariance=full_covariance,
        )

    def _calculate_posterior_distribution(
        self,
        parameters: GPBaseParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate the posterior distribution of the Gaussian Processes.
            - n is the number of training pairs in x_train and y_train
            - m is the number of points in x
            - d is the number of input dimensions
            - k is the number of output dimensions

        Args:
            parameters: parameters of the kernel
            x_train: training design matrix of shape (n, d)
            y_train: training response matrix of shape (n, k)
            x: design matrix of shape (m, d)

        Returns: the mean and covariance of the posterior distribution

        """
        number_of_train_points = x_train.shape[0]
        gram_train = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x1=x_train,
            x2=x_train,
            full_covariance=True,
        )
        gram_train_x = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x1=x_train,
            x2=x,
            full_covariance=True,
        )
        gram_x = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x1=x,
            x2=x,
            full_covariance=True,
        )
        observation_noise_matrix = jnp.eye(number_of_train_points) * jnp.exp(
            parameters.log_observation_noise
        )
        cholesky_decomposition_and_lower = jsp.linalg.cho_factor(
            gram_train + observation_noise_matrix
        )
        kernel_mean = gram_train_x.T @ jsp.linalg.cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=y_train
        )
        mean = kernel_mean + self.mean.predict(parameters.mean, x)
        covariance = gram_x - gram_train_x.T @ jsp.linalg.cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_x
        )
        return mean, covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_posterior_distribution(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> Gaussian:
        """
        Calculate the posterior distribution of the Gaussian Processes.
            - n is the number of training pairs in x_train and y_train
            - m is the number of points in x
            - d is the number of input dimensions
            - k is the number of output dimensions

        Args:
            parameters: parameters of the kernel
            x_train: training design matrix of shape (n, d)
            y_train: training response matrix of shape (n, k)
            x: design matrix of shape (m, d)
            full_covariance: whether to compute the full covariance matrix or just the diagonal (requires m1 == m2)

        Returns: the prior distribution

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        if full_covariance:
            mean, covariance = self._calculate_posterior_distribution(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x,
            )
        else:
            mean, covariance = jax.vmap(
                lambda x_: self._calculate_posterior_distribution(
                    parameters=parameters,
                    x_train=x_train,
                    y_train=y_train,
                    x=x_,
                )
            )(x[:, None, ...])
        return Gaussian(
            mean=mean,
            covariance=covariance,
            full_covariance=False,
        )
