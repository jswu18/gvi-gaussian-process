from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit
from jax.scipy.linalg import cho_factor, cho_solve

from src.kernels.reference_kernels import Kernel


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
                reference_gaussian_measure_parameters["kernel"], x, y
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
        regularisation: float = 1e-5,
    ):
        super().__init__(reference_gaussian_measure_parameters, reference_kernel)
        self.inducing_points = inducing_points
        self.regularisation = regularisation
        self.kzz = self.reference_kernel_gram(inducing_points, inducing_points)
        self.kzz_cholesky_decomposition_and_lower = cho_factor(
            self.kzz + regularisation * jnp.eye(self.kzz.shape[0])
        )

    @staticmethod
    def _calculate_cholesky(
        kzz: jnp.ndarray,
        kxz: jnp.ndarray,
        reference_gaussian_measure_observation_precision: float,
        regularisation: float,
    ) -> jnp.ndarray:
        cholesky_ = jnp.linalg.inv(
            jnp.linalg.cholesky(
                kzz
                + reference_gaussian_measure_observation_precision * kxz.T @ kxz
                + regularisation * jnp.eye(kzz.shape[0])
            )
        )
        return jnp.linalg.cholesky(cholesky_.T @ cholesky_)

    def gram(
        self, x: jnp.ndarray, y: jnp.ndarray = None, parameters: FrozenDict = None
    ) -> jnp.ndarray:
        kxz = self.reference_kernel_gram(
            self.reference_gaussian_measure_parameters["kernel"],
            x,
            self.inducing_points,
        )
        reference_gaussian_measure_observation_precision = 1 / (
            jnp.exp(self.reference_gaussian_measure_parameters["log_observation_noise"])
            ** 2
        )
        cholesky_el_matrix_x = self._calculate_cholesky(
            kzz=self.kzz,
            kxz=kxz,
            reference_gaussian_measure_observation_precision=reference_gaussian_measure_observation_precision,
            regularisation=self.regularisation,
        )
        if y is None:
            y = x
            kyz = kxz
            cholesky_el_matrix_y = cholesky_el_matrix_x
        else:
            kyz = self.reference_kernel_gram(
                self.reference_gaussian_measure_parameters["kernel"],
                y,
                self.inducing_points,
            )
            cholesky_el_matrix_y = self._calculate_cholesky(
                kzz=self.kzz,
                kxz=kyz,
                reference_gaussian_measure_precision=self.reference_gaussian_measure_parameters[
                    "precision"
                ],
                regularisation=self.regularisation,
            )

        kxy = self.reference_kernel_gram(
            self.reference_gaussian_measure_parameters["kernel"], x, y
        )
        sigma_matrix = cholesky_el_matrix_x.T @ cholesky_el_matrix_y
        return (
            kxy
            - kxz
            @ cho_solve(c_and_lower=self.kzz_cholesky_decomposition_and_lower, b=kyz.T)
            + kxz @ sigma_matrix @ kyz.T
        )
