from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import optax
from jax import grad
from jax.scipy.stats import multivariate_normal
from optax import GradientTransformation

from src.kernels.kernels import Kernel
from src.parameters import Parameters


@dataclass
class GaussianMeasureParameters(Parameters, ABC):
    """
    An abstract dataclass containing parameters for a Gaussian measure.
    """


class GaussianMeasure(ABC):
    """
    An abstract Gaussian measure.
    """

    Parameters: GaussianMeasureParameters = None


@dataclass
class GaussianProcessParameters(GaussianMeasureParameters):
    """
    Parameters for a Gaussian Process:
        log_sigma: logarithm of the noise parameter
        kernel: parameters for the chosen kernel
    """

    log_sigma: float
    kernel: Dict[str, Any]

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @sigma.setter
    def sigma(self, value: float) -> None:
        self.log_sigma = jnp.log(value)


class GaussianProcess(GaussianMeasure):
    """
    A Gaussian measure defined with a kernel, better known as a Gaussian Process.
    """

    Parameters = GaussianProcessParameters

    def __init__(self, kernel: Kernel, x: jnp.ndarray, y: jnp.ndarray) -> None:
        """Initialising requires a kernel and data to condition the distribution.

        Args:
            kernel: kernel for the Gaussian Process
            x: design matrix (number_of_features, number_of_dimensions)
            y: response vector (number_of_features, )
        """
        self.number_of_train_points = x.shape[0]
        self.x = x
        self.y = y
        self.kernel = kernel

    def posterior_distribution(
        self, x: jnp.ndarray, **parameter_args
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the posterior distribution for test points x.

        Args:
            x: test points (number_of_features, number_of_dimensions)
            **parameter_args: parameter arguments for the kernel

        Returns:
            mean: the distribution mean (number_of_features, )
            covariance: the distribution covariance (number_of_features, number_of_features)
        """
        parameters = self.Parameters(**parameter_args)
        kxy = self.kernel(self.x, x, **parameters.kernel)
        kyy = self.kernel(x, **parameters.kernel)
        kxx = self.kernel(self.x, **parameters.kernel)
        inv_kxx_shifted = jnp.linalg.inv(
            kxx + parameters.variance * jnp.eye(self.number_of_train_points)
        )

        mean = (kxy.T @ inv_kxx_shifted @ self.y).reshape(
            -1,
        )
        covariance = kyy - kxy.T @ inv_kxx_shifted @ kxy
        return mean, covariance

    def posterior_negative_log_likelihood(
        self, x: jnp.ndarray, y: jnp.ndarray, **parameter_args
    ) -> jnp.float64:
        """The negative log likelihood of the posterior distribution for test data (x, y).

        Args:
            x: test points (number_of_features, number_of_dimensions)
            y: test point responses (number_of_features, )
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            The negative log likelihood.
        """
        mean, covariance = self.posterior_distribution(x, **parameter_args)
        return -jnp.sum(
            multivariate_normal.logpdf(x=y.reshape(1, -1), mean=mean, cov=covariance)
        )

    def _compute_grad(
        self, x: jnp.ndarray, y: jnp.ndarray, **parameter_args
    ) -> Dict[str, Any]:
        """Calculate the gradient of the posterior negative log likelihood with respect to the parameters.

        Args:
            x: test points (number_of_features, number_of_dimensions)
            y: test point responses (number_of_features, )
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            A dictionary of the gradients for each parameter argument.
        """
        gradients = grad(
            lambda params: self.posterior_negative_log_likelihood(x, y, **params)
        )(parameter_args)
        return gradients

    def train(
        self,
        parameters: GaussianProcessParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
        optimizer: GradientTransformation,
        number_of_training_iterations: int,
    ) -> GaussianMeasureParameters:
        """Train the parameters for a Gaussian Process by optimising the negative log likelihood.

        Args:
            parameters: parameters dataclass for the Gaussian Process
            x: test points (number_of_features, number_of_dimensions)
            y: test point responses (number_of_features, )
            optimizer: jax optimizer object
            number_of_training_iterations: number of iterations to perform the optimizer

        Returns:
            A parameters dataclass containing the optimised parameters.
        """
        parameter_args = asdict(parameters)
        opt_state = optimizer.init(parameter_args)
        for _ in range(number_of_training_iterations):
            gradients = self._compute_grad(x, y, **parameter_args)
            updates, opt_state = optimizer.update(gradients, opt_state)
            parameter_args = optax.apply_updates(parameter_args, updates)
        return self.Parameters(**parameter_args)
