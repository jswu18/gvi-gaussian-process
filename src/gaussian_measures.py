from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import jax.scipy
import optax
from jax import grad
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

    def _compute_kxx_shifted_cholesky_decomposition(
        self, parameters
    ) -> Tuple[jnp.ndarray, bool]:
        """
        Cholesky decomposition of (kxx + (1/σ^2)*I)

        Args:
            parameters: parameters dataclass for the Gaussian Process

        Returns:
            cholesky_decomposition_kxx_shifted: the cholesky decomposition (number_of_features, number_of_features)
            lower_flag: flag indicating whether the factor is in the lower or upper triangle
        """
        kxx = self.kernel(self.x, **parameters.kernel)
        kxx_shifted = kxx + parameters.variance * jnp.eye(self.number_of_train_points)
        kxx_shifted_cholesky_decomposition, lower_flag = jax.scipy.linalg.cho_factor(
            a=kxx_shifted, lower=True
        )
        return kxx_shifted_cholesky_decomposition, lower_flag

    def posterior_distribution(
        self, x: jnp.ndarray, **parameter_args
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the posterior distribution for test points x.
        Reference: http://gaussianprocess.org/gpml/chapters/RW2.pdf

        Args:
            x: test points (number_of_features, number_of_dimensions)
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            mean: the distribution mean (number_of_features, )
            covariance: the distribution covariance (number_of_features, number_of_features)
        """
        parameters = self.Parameters(**parameter_args)
        kxy = self.kernel(self.x, x, **parameters.kernel)
        kyy = self.kernel(x, **parameters.kernel)
        (
            kxx_shifted_cholesky_decomposition,
            lower_flag,
        ) = self._compute_kxx_shifted_cholesky_decomposition(parameters)

        mean = (
            kxy.T
            @ jax.scipy.linalg.cho_solve(
                c_and_lower=(kxx_shifted_cholesky_decomposition, lower_flag), b=self.y
            )
        ).reshape(
            -1,
        )
        covariance = kyy - kxy.T @ jax.scipy.linalg.cho_solve(
            (kxx_shifted_cholesky_decomposition, lower_flag), kxy
        )
        return mean, covariance

    def posterior_negative_log_likelihood(self, **parameter_args) -> jnp.float64:
        """The negative log likelihood of the posterior distribution for the training data (x, y).
        Reference: http://gaussianprocess.org/gpml/chapters/RW2.pdf

        Args:
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            The negative log likelihood.
        """
        parameters = self.Parameters(**parameter_args)
        (
            kxx_shifted_cholesky_decomposition,
            lower_flag,
        ) = self._compute_kxx_shifted_cholesky_decomposition(parameters)

        negative_log_likelihood = -(
            -0.5
            * (
                self.y.T
                @ jax.scipy.linalg.cho_solve(
                    c_and_lower=(kxx_shifted_cholesky_decomposition, lower_flag),
                    b=self.y,
                )
            )
            - jnp.trace(jnp.log(kxx_shifted_cholesky_decomposition))
            - (self.number_of_train_points / 2) * jnp.log(2 * jnp.pi)
        )
        return negative_log_likelihood

    def _compute_gradient(self, **parameter_args) -> Dict[str, Any]:
        """Calculate the gradient of the posterior negative log likelihood with respect to the parameters.

        Args:
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            A dictionary of the gradients for each parameter argument.
        """
        gradients = grad(
            lambda params: self.posterior_negative_log_likelihood(**params)
        )(parameter_args)
        return gradients

    def train(
        self,
        optimizer: GradientTransformation,
        number_of_training_iterations: int,
        **parameter_args,
    ) -> GaussianMeasureParameters:
        """Train the parameters for a Gaussian Process by optimising the negative log likelihood.

        Args:
            optimizer: jax optimizer object
            number_of_training_iterations: number of iterations to perform the optimizer
            **parameter_args: parameter arguments for the Gaussian Process

        Returns:
            A parameters dataclass containing the optimised parameters.
        """
        opt_state = optimizer.init(parameter_args)
        for _ in range(number_of_training_iterations):
            gradients = self._compute_gradient(**parameter_args)
            updates, opt_state = optimizer.update(gradients, opt_state)
            parameter_args = optax.apply_updates(parameter_args, updates)
        return self.Parameters(**parameter_args)
