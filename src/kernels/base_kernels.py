from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.kernels.kernels import Kernel, KernelParameters


@dataclass
class BaseKernelParameters(KernelParameters, ABC):
    """
    Abstract dataclass containing parameters for a standard base kernel.
    """


class BaseKernel(Kernel, ABC):
    """
    Abstract kernel class for base kernels that are not reliant on other kernels and each
    element of the gram matrix can be calculated independently.
    """

    Parameters: BaseKernelParameters = None

    @abstractmethod
    def _kernel(
        self, parameters: BaseKernelParameters, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Kernel evaluation between a single feature x and a single feature y.

        Args:
            parameters: parameters dataclass for the kernel
            x: ndarray of shape (number_of_dimensions,)
            y: ndarray of shape (number_of_dimensions,)

        Returns:
            The kernel evaluation. (1, 1)
        """
        raise NotImplementedError

    def kernel(
        self, parameters: BaseKernelParameters, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Kernel evaluation for an arbitrary number of x features and y features. Compute k(x, x) if y is None.
        This method requires the parameters dataclass and is better suited for parameter optimisation.

        Args:
            parameters: parameters dataclass for the kernel
            x: ndarray of shape (number_of_x_features, number_of_dimensions)
            y: ndarray of shape (number_of_y_features, number_of_dimensions)

        Returns:
            A gram matrix k(x, y), if y is None then k(x,x). (number_of_x_features, number_of_y_features)
        """
        # compute k(x, x) if y is None
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        assert (
            x.shape[1] == y.shape[1]
        ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"

        return vmap(
            lambda x_i: vmap(
                lambda y_i: self._kernel(parameters, x_i, y_i),
            )(y),
        )(x)

    def diagonal(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        **parameter_args,
    ) -> jnp.ndarray:
        """Kernel evaluation of only the diagonal terms of the gram matrix.

        Args:
            x: ndarray of shape (number_of_x_features, number_of_dimensions)
            y: ndarray of shape (number_of_y_features, number_of_dimensions)
            **parameter_args: parameter arguments for the kernel

        Returns:
            A diagonal of gram matrix k(x, y), if y is None then trace(k(x,x)).
            (number_of_x_features, number_of_y_features)
        """
        # compute k(x, x) if y is None
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        assert (
            x.shape[1] == y.shape[1]
        ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"
        assert (
            x.shape[0] == y.shape[0]
        ), f"Must have same number of features for diagonal: {x.shape[0]=} != {y.shape[0]=}"

        return vmap(
            lambda x_i, y_i: self._kernel(
                parameters=self.Parameters(**parameter_args),
                x=x_i,
                y=y_i,
            ),
        )(x, y)


@dataclass
class GaussianKernelParameters(BaseKernelParameters):
    """
    Parameters for the Gaussian Kernel:
        log_sigma: logarithm of the sigma parameter, the logarithm is used to ensure that sigma >= 0
    """

    log_sigma: float

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @sigma.setter
    def sigma(self, value: float) -> None:
        self.log_sigma = jnp.log(value)


class GaussianKernel(BaseKernel):
    """
    The Gaussian kernel defined as:
        k(x, y) = exp(-σ||x-y||_2^2)
    where σ>0.
    """

    Parameters = GaussianKernelParameters

    def _kernel(
        self,
        parameters: GaussianKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Kernel evaluation between a single feature x and a single feature y.

        Args:
            parameters: parameters dataclass for the Gaussian kernel
            x: ndarray of shape (number_of_dimensions,)
            y: ndarray of shape (number_of_dimensions,)

        Returns:
            The kernel evaluation.
        """
        return jnp.exp(-parameters.sigma * jnp.linalg.norm(x - y, ord=2) ** 2)


@dataclass
class ARDKernelParameters(BaseKernelParameters):
    """
    Parameters for the ARD Kernel:
        log_sigma: logarithm of the sigma parameter, the logarithm is used to ensure that sigma >= 0
        log_alpha: a vector matching the dimension of the data, acting as an independent length scale for each dimension
    """

    log_sigma: float
    log_alpha: jnp.ndarray

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @sigma.setter
    def sigma(self, value) -> None:
        self.log_sigma = jnp.log(value)

    @property
    def alpha(self) -> np.ndarray:
        return jnp.exp(self.log_alpha)

    @alpha.setter
    def alpha(self, value) -> None:
        self.log_alpha = jnp.log(value)

    @property
    def precision_matrix(self) -> np.ndarray:
        return jnp.diag(1 / jnp.square(self.alpha))


class ARDKernel(BaseKernel):
    """
    The ARD Kernel defined as:
        k(x, y) = (σ_f)^2 * exp( -0.5 * sum_d(((x_d - y_d)^2)/(α_d)^2) )
    where σ_f>0, the kernel scaling factor and α_d the length scale for dimension d
    """

    Parameters = ARDKernelParameters

    def _kernel(
        self, parameters: ARDKernelParameters, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Kernel evaluation between a single feature x and a single feature y.

        Args:
            parameters: parameters dataclass for the ARD kernel
            x: ndarray of shape (number_of_dimensions,)
            y: ndarray of shape (number_of_dimensions,)

        Returns:
            The kernel evaluation.
        """
        assert (
            x.shape[0] == parameters.alpha.shape[0]
        ), f"Dimension Mismatch: {x.shape[0]=} != {parameters.alpha.shape[0]=}"

        return jnp.sum(
            parameters.variance
            * jnp.exp(-0.5 * (x - y).T @ parameters.precision_matrix @ (x - y))
        )
