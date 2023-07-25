from typing import Any, Dict, Literal, Tuple, Union

import jax.numpy as jnp
import pydantic
from flax.core import FrozenDict

from src.gps.base.base import GPBase, GPBaseParameters
from src.kernels import TemperedKernel, TemperedKernelParameters
from src.means.base import MeanBaseParameters
from src.utils.custom_types import JaxFloatType
from src.utils.jit_compiler import JitCompiler

PRNGKey = Any  # pylint: disable=invalid-name


class TemperedGPParameters(GPBaseParameters):
    log_observation_noise: JaxFloatType = None
    mean: MeanBaseParameters = None
    kernel: TemperedKernelParameters


class TemperedGP(GPBase):
    Parameters = TemperedGPParameters

    def __init__(self, base_gp: GPBase, base_gp_parameters: GPBaseParameters):
        self.base_gp = base_gp
        self.tempered_base_gp = base_gp
        self.tempered_base_gp.kernel = TemperedKernel(
            base_kernel=self.base_gp.kernel,
            base_kernel_parameters=base_gp_parameters.kernel,
            number_output_dimensions=base_gp.kernel.number_output_dimensions,
        )
        self.base_gp_parameters = base_gp_parameters
        super().__init__(
            mean=self.tempered_base_gp.mean,
            kernel=self.tempered_base_gp.kernel,
        )
        self._jit_compiled_predict_probability = JitCompiler(self._predict_probability)

    def _construct_tempered_base_gp_parameters(
        self, parameters: Union[FrozenDict, Dict, TemperedGPParameters]
    ) -> GPBaseParameters:
        if not isinstance(parameters, self.Parameters):
            parameters = self.generate_parameters(parameters)
        return self.tempered_base_gp.Parameters(
            log_observation_noise=self.base_gp_parameters.log_observation_noise,
            mean=self.base_gp_parameters.mean,
            kernel=parameters.kernel,
        )

    def _calculate_prediction_gaussian(
        self,
        parameters: TemperedGPParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_prediction_gaussian(
            parameters=tempered_base_gp_parameters,
            x=x,
            full_covariance=full_covariance,
        )

    def _calculate_prediction_gaussian_covariance(
        self,
        parameters: TemperedGPParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_prediction_gaussian_covariance(
            parameters=tempered_base_gp_parameters,
            x=x,
            full_covariance=full_covariance,
        )

    def _predict_probability(
        self, parameters: TemperedGPParameters, x: jnp.ndarray
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._predict_probability(
            parameters=tempered_base_gp_parameters,
            x=x,
        )

    def _construct_distribution(
        self,
        probabilities: Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ):
        return self.tempered_base_gp._construct_distribution(
            probabilities=probabilities,
        )

    def _calculate_prior_covariance(
        self,
        parameters: TemperedGPParameters,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_prior_covariance(
            parameters=tempered_base_gp_parameters,
            x=x,
            full_covariance=full_covariance,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def calculate_prior(
        self,
        parameters: Union[Dict, FrozenDict, TemperedGPParameters],
        x: jnp.ndarray,
        full_covariance: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp.calculate_prior(
            parameters=tempered_base_gp_parameters,
            x=x,
            full_covariance=full_covariance,
        )

    def _calculate_partial_posterior_covariance(
        self,
        parameters: TemperedGPParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_partial_posterior_covariance(
            parameters=tempered_base_gp_parameters,
            x_train=x_train,
            y_train=y_train,
            x=x,
        )

    def _calculate_full_posterior_covariance(
        self,
        parameters: TemperedGPParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_full_posterior_covariance(
            parameters=tempered_base_gp_parameters,
            x_train=x_train,
            y_train=y_train,
            x=x,
        )

    def _calculate_partial_posterior(
        self,
        parameters: TemperedGPParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_partial_posterior(
            parameters=tempered_base_gp_parameters,
            x_train=x_train,
            y_train=y_train,
            x=x,
        )

    def _calculate_full_posterior(
        self,
        parameters: TemperedGPParameters,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        tempered_base_gp_parameters = self._construct_tempered_base_gp_parameters(
            parameters
        )
        return self.tempered_base_gp._calculate_full_posterior(
            parameters=tempered_base_gp_parameters,
            x_train=x_train,
            y_train=y_train,
            x=x,
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate_parameters(
        self, parameters: Union[FrozenDict, Dict]
    ) -> TemperedGPParameters:
        return TemperedGP.Parameters(
            kernel=parameters["kernel"],
        )

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def initialise_random_parameters(
        self,
        key: PRNGKey,
    ) -> TemperedGPParameters:
        pass
