from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.reference_kernels import Kernel

PRNGKey = Any  # pylint: disable=invalid-name


class ApproximationKernel(Kernel, ABC):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_kernel: Kernel,
    ):
        self.reference_gaussian_measure_parameters = (
            reference_gaussian_measure_parameters
        )
        self.reference_kernel_gram = jit(
            lambda x, y=None: reference_kernel.gram(
                parameters=reference_gaussian_measure_parameters["kernel"],
                x=x,
                y=y,
            )
        )

    @abstractmethod
    def gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        raise NotImplementedError


class StochasticVariationalGaussianProcessKernel(ApproximationKernel):
    def __init__(
        self,
        reference_gaussian_measure_parameters: FrozenDict,
        reference_kernel: Kernel,
        inducing_points: jnp.ndarray,
        training_points: jnp.ndarray,
        regularisation: float = 1e-5,
    ):
        super().__init__(reference_gaussian_measure_parameters, reference_kernel)
        self.inducing_points = inducing_points
        self.training_points = training_points
        self.regularisation = regularisation
        self.kzz = self.reference_kernel_gram(x=inducing_points)
        self.kzz_cholesky_decomposition_and_lower = cho_factor(
            self.kzz + regularisation * jnp.eye(self.kzz.shape[0])
        )
        self.sigma_matrix = self._calculate_sigma_matrix(
            gram_inducing=self.kzz,
            gram_inducing_train=self.reference_kernel_gram(
                x=inducing_points, y=training_points
            ),
            reference_gaussian_measure_observation_precision=1
            / jnp.exp(reference_gaussian_measure_parameters["log_observation_noise"]),
            regularisation=regularisation,
        )

    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> FrozenDict:
        pass

    def initialise_parameters(self, **kwargs) -> FrozenDict:
        pass

    @staticmethod
    def _calculate_sigma_matrix(
        gram_inducing: jnp.ndarray,
        gram_inducing_train: jnp.ndarray,
        reference_gaussian_measure_observation_precision: float,
        regularisation: float,
    ) -> jnp.ndarray:
        cholesky_decomposition_and_lower = cho_factor(
            jnp.linalg.cholesky(
                gram_inducing
                + reference_gaussian_measure_observation_precision
                * gram_inducing_train
                @ gram_inducing_train.T
                + regularisation * jnp.eye(gram_inducing.shape[0])
            )
        )
        el_matrix = cho_solve(
            c_and_lower=cholesky_decomposition_and_lower,
            b=jnp.eye(gram_inducing.shape[0]),
        )
        return el_matrix @ el_matrix.T

    def gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        kxz = self.reference_kernel_gram(
            x=x,
            y=self.inducing_points,
        )
        if y is None:
            y = x
            kyz = kxz
        else:
            kyz = self.reference_kernel_gram(
                x=y,
                y=self.inducing_points,
            )

        kxy = self.reference_kernel_gram(x=x, y=y)
        return (
            kxy
            - kxz
            @ cho_solve(c_and_lower=self.kzz_cholesky_decomposition_and_lower, b=kyz.T)
            + kxz @ self.sigma_matrix @ kyz.T
        )
