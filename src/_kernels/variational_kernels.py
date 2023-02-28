from dataclasses import asdict, dataclass

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from src._kernels.base_kernels import BaseKernel, BaseKernelParameters
from src._kernels.kernels import Kernel, KernelParameters


@dataclass
class VariationalKernelParameters(KernelParameters):
    log_sigma: float
    base_kernel_parameters: BaseKernelParameters

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @sigma.setter
    def sigma(self, value: float) -> None:
        self.log_sigma = jnp.log(value)

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def precision(self) -> float:
        return 1 / self.variance


class VariationalKernel(Kernel):

    Parameters = VariationalKernelParameters

    def __init__(
        self,
        base_kernel: BaseKernel,
        inducing_points: jnp.ndarray,
        regularisation: float = 1e-5,
    ) -> None:
        self.base_kernel = base_kernel
        self.inducing_points = inducing_points
        self.regularisation = regularisation

    @staticmethod
    def _calculate_cholesky(
        parameters: VariationalKernelParameters,
        kzz: jnp.ndarray,
        kxz: jnp.ndarray,
        regularisation: float,
    ) -> jnp.ndarray:
        cholesky_ = jnp.linalg.inv(
            jnp.linalg.cholesky(
                kzz
                + parameters.precision * kxz.T @ kxz
                + regularisation * jnp.eye(kzz.shape[0])
            )
        )
        return jnp.linalg.cholesky(cholesky_.T @ cholesky_)

    def kernel(
        self,
        parameters: VariationalKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if isinstance(parameters.base_kernel_parameters, dict):
            parameters.base_kernel_parameters = self.base_kernel.Parameters(
                **parameters.base_kernel_parameters
            )
        kzz = self.base_kernel(
            self.inducing_points, **asdict(parameters.base_kernel_parameters)
        )
        cholesky_kzz_and_lower = cho_factor(
            kzz + self.regularisation * jnp.eye(kzz.shape[0])
        )

        kxz = self.base_kernel(
            x, self.inducing_points, **asdict(parameters.base_kernel_parameters)
        )
        cholesky_x = self._calculate_cholesky(
            parameters=parameters, kzz=kzz, kxz=kxz, regularisation=self.regularisation
        )

        # compute k(x, x) if y is None
        if y is None:
            y = x
            kyz = kxz
            cholesky_y = cholesky_x
        else:
            kyz = self.base_kernel(
                y,
                self.inducing_points,
                **asdict(parameters.base_kernel_parameters),
            )
            cholesky_y = self._calculate_cholesky(
                parameters=parameters,
                kzz=kzz,
                kxz=kyz,
                regularisation=self.regularisation,
            )

        kxy = self.base_kernel(x, y, **asdict(parameters.base_kernel_parameters))
        covariance = cholesky_x.T @ cholesky_y
        return (
            kxy
            - kxz @ cho_solve(c_and_lower=cholesky_kzz_and_lower, b=kyz.T)
            + kxz @ covariance @ kyz.T
        )
