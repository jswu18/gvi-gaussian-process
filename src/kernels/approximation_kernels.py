from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.reference_kernels import Kernel
from src.utils import add_diagonal_regulariser

PRNGKey = Any  # pylint: disable=invalid-name


class ApproximationKernel(Kernel, ABC):
    """
    Approximation kernels which are defined with respect to a reference Gaussian measure.
    """

    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_kernel: Kernel,
    ):
        """
        Defining the kernel with respect to a reference Gaussian measure.

        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_kernel: the kernel of the reference Gaussian measure.
        """
        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )

        # define a jit-compiled version of the reference kernel gram matrix using the reference kernel parameters
        self.calculate_reference_gram = jit(
            lambda x, y=None: reference_kernel.calculate_gram(
                parameters=reference_gaussian_measure_parameters["kernel"],
                x=x,
                y=y,
            )
        )

    @abstractmethod
    def calculate_gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        """
        Computes the Gram matrix of the kernel. If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)

        """
        raise NotImplementedError


class StochasticVariationalGaussianProcessKernel(ApproximationKernel):
    """
    The stochastic variational Gaussian process kernel as defined in Titsias (2009).
    """

    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_kernel: Kernel,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        diagonal_regularisation: float = 1e-5,
        is_diagonal_regularisation_absolute_scale: bool = False,
    ):
        """
        Defining the stochastic variational Gaussian process kernel using the reference Gaussian measure
        and inducing points.
        Args:
            reference_gaussian_measure_parameters: the parameters of the reference Gaussian measure.
            reference_kernel: the kernel of the reference Gaussian measure.
            inducing_points: the inducing points of the stochastic variational Gaussian process.
            training_points: the training points of the stochastic variational Gaussian process.
            diagonal_regularisation: the diagonal regularisation used to stabilise the Cholesky decomposition.
            is_diagonal_regularisation_absolute_scale: whether the diagonal regularisation is an absolute scale.
        """
        super().__init__(reference_gaussian_measure_parameters, reference_kernel)
        self.inducing_points = inducing_points
        self.training_points = training_points
        self.diagonal_regularisation = diagonal_regularisation
        self.is_diagonal_regularisation_absolute_scale = (
            is_diagonal_regularisation_absolute_scale
        )
        self.reference_gram_inducing = self.calculate_reference_gram(x=inducing_points)
        self.reference_gram_inducing_cholesky_decomposition_and_lower = cho_factor(
            add_diagonal_regulariser(
                matrix=self.reference_gram_inducing,
                diagonal_regularisation=diagonal_regularisation,
                is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
            )
        )
        self.sigma_matrix = self._calculate_sigma_matrix(
            gram_inducing=self.reference_gram_inducing,
            gram_inducing_train=self.calculate_reference_gram(
                x=inducing_points, y=training_points
            ),
            reference_gaussian_measure_observation_precision=1
            / jnp.exp(reference_gaussian_measure_parameters["log_observation_noise"]),
            diagonal_regularisation=diagonal_regularisation,
            is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
        )

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        """
        Initialise the parameters of the module using a random key.
        Args:
            key: A random key used to initialise the parameters.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        """
        Initialise the parameters of the module using the provided arguments.
        Args:
            **kwargs: The parameters of the module.

        Returns: A dictionary of the parameters of the module.

        """
        pass

    @staticmethod
    def _calculate_sigma_matrix(
        gram_inducing: jnp.ndarray,
        gram_inducing_train: jnp.ndarray,
        reference_gaussian_measure_observation_precision: float,
        diagonal_regularisation: float,
        is_diagonal_regularisation_absolute_scale: bool,
    ) -> jnp.ndarray:
        cholesky_decomposition_and_lower = cho_factor(
            jnp.linalg.cholesky(
                add_diagonal_regulariser(
                    matrix=(
                        gram_inducing
                        + reference_gaussian_measure_observation_precision
                        * gram_inducing_train
                        @ gram_inducing_train.T
                    ),
                    diagonal_regularisation=diagonal_regularisation,
                    is_diagonal_regularisation_absolute_scale=is_diagonal_regularisation_absolute_scale,
                )
            )
        )
        el_matrix = cho_solve(
            c_and_lower=cholesky_decomposition_and_lower,
            b=jnp.eye(gram_inducing.shape[0]),
        )
        return el_matrix @ el_matrix.T

    def calculate_gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        """
        Computing the Gram matrix using for the SVGP which depends on the reference kernel.
        If y is None, the Gram matrix is computed for x and x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: the kernel gram matrix of shape (n, m)

        """
        reference_gram_x_inducing = self.calculate_reference_gram(
            x=x,
            y=self.inducing_points,
        )
        if y is None:
            y = x
            reference_gram_y_inducing = reference_gram_x_inducing
        else:
            reference_gram_y_inducing = self.calculate_reference_gram(
                x=y,
                y=self.inducing_points,
            )

        reference_gram_x_y = self.calculate_reference_gram(x=x, y=y)
        return (
            reference_gram_x_y
            - reference_gram_x_inducing
            @ cho_solve(
                c_and_lower=self.reference_gram_inducing_cholesky_decomposition_and_lower,
                b=reference_gram_y_inducing.T,
            )
            + reference_gram_x_inducing
            @ self.sigma_matrix
            @ reference_gram_y_inducing.T
        )
