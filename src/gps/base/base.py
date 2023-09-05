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
from src.module import PYDANTIC_VALIDATION_CONFIG, Module, ModuleParameters
from src.utils.custom_types import JaxFloatType


class GPBaseParameters(ModuleParameters, ABC):
    """
    Parameters for a Gaussian process defined with respect to a mean function and a kernel.
    """

    log_observation_noise: JaxFloatType
    mean: MeanBaseParameters
    kernel: KernelBaseParameters


class GPBase(Module, ABC):
    """
    A Gaussian process defined with respect to a mean function and a kernel.
    """

    Parameters = GPBaseParameters

    # indicates the type of distribution that is returned by the predict method
    PredictDistribution = Distribution

    def __init__(
        self,
        mean: MeanBase,
        kernel: KernelBase,
    ):
        """
        Defining the mean function, and the kernel for the Gaussian process.

        Args:
            mean: the mean function of the Gaussian measure
            kernel: the kernel of the Gaussian measure
        """
        self.mean = mean
        self.kernel = kernel
        self._jit_compiled_predict_probability = jax.jit(
            lambda parameters, x: self._predict_probability(parameters=parameters, x=x)
        )
        super().__init__(preprocess_function=None)

    @abstractmethod
    def _calculate_prediction_gaussian(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the mean and covariance of the Gaussian distribution of the prediction.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made
            full_covariance: whether the full covariance matrix is returned or just the diagonal

        Returns: the mean and covariance of the Gaussian distribution of the prediction

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        """
        Calculates only the covariance of the Gaussian distribution of the prediction.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made
            full_covariance: whether the full covariance matrix is returned or just the diagonal

        Returns: the covariance of the Gaussian distribution of the prediction

        """
        raise NotImplementedError

    @abstractmethod
    def _predict_probability(
        self, parameters: GPBaseParameters, x: jnp.ndarray
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        If the Gaussian process is used as a classifier, this method returns the probabilities of the labels.
        If the Gaussian process is used as a regressor, this method returns the mean and covariance of the
        Gaussian distribution of the prediction.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made

        Returns: the probabilities of the labels or the mean and covariance of the Gaussian distribution of the
                 prediction

        """
        raise NotImplementedError

    @abstractmethod
    def _construct_distribution(
        self,
        probabilities: Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> Distribution:
        """
        Constructs the distribution of the prediction.
        Args:
            probabilities: the probabilities of the labels or the mean and covariance of the Gaussian distribution of
                            the prediction

        Returns: the distribution of the prediction (Gaussian or Multinomial)

        """
        raise NotImplementedError

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def predict_probability(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
    ) -> Distribution:
        """
        If the Gaussian process is used as a classifier, this method returns the probabilities of the labels.
        If the Gaussian process is used as a regressor, this method returns the mean and covariance of the
        Gaussian distribution of the prediction.
        This method is a wrapper for the _predict_probability method to run the jit-compiled version of the method.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made

        Returns: the probabilities of the labels or the mean and covariance of the Gaussian distribution of the
                    prediction

        """
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        Module.check_parameters(parameters, self.Parameters)
        probabilities = self._jit_compiled_predict_probability(parameters.dict(), x)
        return self._construct_distribution(probabilities)

    def construct_observation_noise_matrix(
        self, log_observation_noise: Union[jnp.ndarray, float], number_of_points: int
    ):
        """
        Constructs the observation noise matrix.
        Args:
            log_observation_noise: the log of the observation noise
            number_of_points: the number of points for which the observation noise matrix is constructed

        Returns: the observation noise matrix of shape (k, n, n) where k is the number of output dimensions and n is
                    the number of points

        """
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
        """
        Calculates the prior covariance matrix.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made
            full_covariance: whether the full covariance matrix is returned or just the diagonal

        Returns: the prior covariance matrix of shape (k, n, n) where k is the number of output dimensions and n is
                    the number of points if full_covariance is True, otherwise the prior covariance matrix of shape
                    (k, n) where k is the number of output dimensions and n is the number of points

        """
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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_prior_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> jnp.ndarray:
        """
        Calculates the prior covariance matrix.
        This method is a wrapper for the _calculate_prior_covariance method to run the jit-compiled version of the
        method.
        Args:
            parameters: the parameters of the Gaussian process
            x: the input points for which the prediction is made
            full_covariance: whether the full covariance matrix is returned or just the diagonal

        Returns: the prior covariance matrix of shape (k, n, n) where k is the number of output dimensions and n is
                    the number of points if full_covariance is True, otherwise the prior covariance matrix of shape
                    (k, n) where k is the number of output dimensions and n is the number of points

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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
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
        """
        Calculate the posterior distribution of the Gaussian Processes.

        Args:
            parameters: the parameters of the Gaussian process
            x_train: the input points of the training data
            y_train: the output points of the training data
            x: the input points for which the prediction is made

        Returns: the posterior covariance matrix

        """
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
        """
        Calculate the posterior distribution of the Gaussian Processes.
        Args:
            parameters: parameters of the kernel
            x_train: training design matrix of shape (n, d)
            y_train: training response matrix of shape (n, k)
            x: design matrix of shape (m, d)

        Returns: the mean (k, m) and covariance (k, m) of the posterior distribution

        """
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
        """
        Calculate the posterior mean and covariance of the Gaussian Processes.

        Args:
            gram_train: the gram matrix of the training points
            gram_train_x: the gram matrix between the training points and the test points
            gram_x: the gram matrix of the test points
            observation_noise_matrix: the observation noise matrix
            y_train: the training response matrix

        Returns: the mean and covariance of the posterior distribution

        """
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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_prediction_gaussian(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Gaussian:
        """
        Calculate the predictive distribution of the Gaussian Processes.
        Args:
            parameters: parameters of the Gaussian Processes
            x: the input design matrix of shape (m, d)
            full_covariance: whether to return the full covariance matrix or just the diagonal

        Returns:

        """
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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_prediction_gaussian_covariance(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        """
        Calculate the predictive distribution of the Gaussian Processes.
        Args:
            parameters: parameters of the Gaussian Processes
            x: the input design matrix of shape (m, d)
            full_covariance: whether to return the full covariance matrix or just the diagonal

        Returns: the predictive covariance matrix of shape (m, m)

        """
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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
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

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
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
