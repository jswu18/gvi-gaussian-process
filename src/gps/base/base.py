from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

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
    def _calculate_prediction_gaussian(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _predict_probability(
        self, parameters: GPBaseParameters, x: jnp.ndarray
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _construct_distribution(
        self,
        probabilities: Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
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
        return self._construct_distribution(probabilities)

    def construct_observation_noise_matrix(
        self, log_observation_noise: Union[jnp.ndarray, float], number_of_points: int
    ):
        return jnp.multiply(
            jnp.atleast_1d(jnp.exp(log_observation_noise))[:, None, None],
            jnp.array(
                [jnp.eye(number_of_points)] * self.kernel.number_output_dimensions
            ),
        )

    def _calculate_prior_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        covariance = self.kernel.calculate_gram(
            parameters=parameters.kernel,
            x1=x,
            x2=x,
            full_covariance=full_covariance,
        )
        if full_covariance:
            # (k, n, n)
            observation_noise_matrix = self.construct_observation_noise_matrix(
                log_observation_noise=parameters.log_observation_noise,
                number_of_points=x.shape[0],
            ).reshape(covariance.shape)
            covariance = covariance + observation_noise_matrix
        else:
            # (k, n)
            covariance = (
                covariance.reshape(self.kernel.number_output_dimensions, x.shape[0])
                + jnp.atleast_1d(jnp.exp(parameters.log_observation_noise))[:, None]
            )
            if self.kernel.number_output_dimensions == 1:
                covariance = covariance.squeeze(axis=0)
        return covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prior_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> jnp.ndarray:
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
        covariance = self._calculate_prior_covariance(
            parameters=parameters,
            x=x,
            full_covariance=full_covariance,
        )
        return covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prior(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        mean = self.mean.predict(parameters.mean, x)
        covariance = self._calculate_prior_covariance(
            parameters=parameters,
            x=x,
            full_covariance=full_covariance,
        )
        return mean, covariance

    def _calculate_partial_posterior_covariance(
        self,
        parameters: GPBaseParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        covariance = jax.vmap(
            lambda x_: self._calculate_full_posterior_covariance(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x_,
            )
        )(x[:, None, ...])
        covariance = jnp.swapaxes(covariance, 0, 1).squeeze(axis=-1).squeeze(axis=-1)
        return covariance

    def _calculate_full_posterior_covariance(
        self,
        parameters: GPBaseParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
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

        Returns: the mean (k, n) and covariance (k, n, n) of the posterior distribution

        """
        number_of_train_points = x_train.shape[0]
        number_of_test_points = x.shape[0]

        # (k, n, n)
        gram_train = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x_train,
                x2=x_train,
                full_covariance=True,
            )
        ).reshape(-1, number_of_train_points, number_of_train_points)

        # (k, n, m)
        gram_train_x = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x_train,
                x2=x,
                full_covariance=True,
            )
        ).reshape(-1, number_of_train_points, number_of_test_points)

        # (k, m, m)
        gram_x = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x,
                x2=x,
                full_covariance=True,
            )
        ).reshape(-1, number_of_test_points, number_of_test_points)

        # (k, n, n)
        observation_noise_matrix = self.construct_observation_noise_matrix(
            log_observation_noise=parameters.log_observation_noise,
            number_of_points=number_of_train_points,
        )
        # (k, n)
        y_train = jnp.atleast_2d(y_train.T)

        # (k, n), (k, n, n)
        kernel_mean, covariance = jax.vmap(
            lambda g_tr, g_tr_x, g_x, obs_noise, y_tr: self.calculate_posterior_matrices_of_kernel(
                gram_train=g_tr,
                gram_train_x=g_tr_x,
                gram_x=g_x,
                observation_noise_matrix=obs_noise,
                y_train=y_tr,
            )
        )(gram_train, gram_train_x, gram_x, observation_noise_matrix, y_train)
        return covariance

    def _calculate_partial_posterior(
        self,
        parameters: GPBaseParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, covariance = jax.vmap(
            lambda x_: self._calculate_full_posterior(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x_,
            )
        )(x[:, None, ...])
        mean = jnp.swapaxes(mean, 0, 1).squeeze(axis=-1)
        covariance = jnp.swapaxes(covariance, 0, 1).squeeze(axis=-1).squeeze(axis=-1)
        return mean, covariance

    def _calculate_full_posterior(
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

        Returns: the mean (k, n) and covariance (k, n, n) of the posterior distribution

        """
        number_of_train_points = x_train.shape[0]
        number_of_test_points = x.shape[0]

        # (k, n, n)
        gram_train = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x_train,
                x2=x_train,
                full_covariance=True,
            )
        ).reshape(-1, number_of_train_points, number_of_train_points)

        # (k, n, m)
        gram_train_x = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x_train,
                x2=x,
                full_covariance=True,
            )
        ).reshape(-1, number_of_train_points, number_of_test_points)

        # (k, m, m)
        gram_x = jnp.atleast_3d(
            self.kernel.calculate_gram(
                parameters=parameters.kernel,
                x1=x,
                x2=x,
                full_covariance=True,
            )
        ).reshape(-1, number_of_test_points, number_of_test_points)

        # (k, n, n)
        observation_noise_matrix = self.construct_observation_noise_matrix(
            log_observation_noise=parameters.log_observation_noise,
            number_of_points=number_of_train_points,
        )

        # (k, n)
        prior_mean = self.mean.predict(parameters.mean, x)

        # (k, n)
        y_train = jnp.atleast_2d(y_train.T)

        # (k, n), (k, n, n)
        kernel_mean, covariance = jax.vmap(
            lambda g_tr, g_tr_x, g_x, obs_noise, y_tr: self.calculate_posterior_matrices_of_kernel(
                gram_train=g_tr,
                gram_train_x=g_tr_x,
                gram_x=g_x,
                observation_noise_matrix=obs_noise,
                y_train=y_tr,
            )
        )(gram_train, gram_train_x, gram_x, observation_noise_matrix, y_train)
        mean = kernel_mean + prior_mean
        return mean, covariance

    @staticmethod
    def calculate_posterior_matrices_of_kernel(
        gram_train: jnp.ndarray,
        gram_train_x: jnp.ndarray,
        gram_x: jnp.ndarray,
        observation_noise_matrix: jnp.ndarray,
        y_train: jnp.ndarray,
    ):
        cholesky_decomposition_and_lower = jsp.linalg.cho_factor(
            gram_train + observation_noise_matrix
        )
        kernel_mean = gram_train_x.T @ jsp.linalg.cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=y_train
        )

        covariance = gram_x - gram_train_x.T @ jsp.linalg.cho_solve(
            c_and_lower=cholesky_decomposition_and_lower, b=gram_train_x
        )
        return kernel_mean, covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prediction_gaussian(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Gaussian:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        mean, covariance = self._calculate_prediction_gaussian(
            parameters=parameters,
            x=x,
            full_covariance=full_covariance,
        )
        return Gaussian(
            mean=mean,
            covariance=covariance,
            full_covariance=full_covariance,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prediction_gaussian_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        covariance = self._calculate_prediction_gaussian_covariance(
            parameters=parameters,
            x=x,
            full_covariance=full_covariance,
        )
        return covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_posterior(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
        full_covariance: bool = True,
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
            full_covariance: whether to compute the full covariance matrix or just the diagonal (requires m1 == m2)

        Returns: the prior distribution

        """
        # convert to Pydantic model if necessary
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        if full_covariance:
            posterior_mean, posterior_covariance = self._calculate_full_posterior(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x,
            )
        else:
            posterior_mean, posterior_covariance = self._calculate_partial_posterior(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x,
            )
        if self.kernel.number_output_dimensions == 1:
            posterior_covariance = posterior_covariance.squeeze(axis=0)
        return posterior_mean, posterior_covariance

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_posterior_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> jnp.ndarray:
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
            posterior_covariance = self._calculate_full_posterior_covariance(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x,
            )
        else:
            posterior_covariance = self._calculate_partial_posterior_covariance(
                parameters=parameters,
                x_train=x_train,
                y_train=y_train,
                x=x,
            )
        if self.kernel.number_output_dimensions == 1:
            posterior_covariance = posterior_covariance.squeeze(axis=0)
        return posterior_covariance
