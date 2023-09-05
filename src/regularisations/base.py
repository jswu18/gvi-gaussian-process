from abc import ABC, abstractmethod
from typing import Dict, Union

import jax
import jax.numpy as jnp
import pydantic
from flax.core.frozen_dict import FrozenDict

from src.distributions import Gaussian
from src.gps.base.base import GPBase, GPBaseParameters
from src.module import PYDANTIC_VALIDATION_CONFIG
from src.regularisations.schemas import RegularisationMode


class RegularisationBase(ABC):
    """
    A base class for all regularisers.
    """

    def __init__(
        self,
        gp: GPBase,
        regulariser: GPBase,
        regulariser_parameters: GPBaseParameters,
        mode: RegularisationMode,
    ):
        self._gp = gp
        self._regulariser = regulariser
        self._regulariser_parameters = regulariser_parameters
        self._mode = mode
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def gp(self) -> GPBase:
        return self._gp

    @gp.setter
    def gp(self, gp: GPBase) -> None:
        """
        Sets the GP to regularise.
        This also recompiles the jitted function.
        Args:
            gp: the GP to regularise

        """
        self._gp = gp
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def regulariser(self) -> GPBase:
        return self._regulariser

    @regulariser.setter
    def regulariser(self, regulariser: GPBase) -> None:
        """
        Sets the regulariser GP.
        This also recompiles the jitted function.
        Args:
            regulariser: the regulariser GP

        """
        self._regulariser = regulariser
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @property
    def regulariser_parameters(self) -> GPBaseParameters:
        return self._regulariser_parameters

    @regulariser_parameters.setter
    def regulariser_parameters(self, regulariser_parameters: GPBaseParameters) -> None:
        """
        Sets the parameters of the regulariser GP.
        This also recompiles the jitted function.
        Args:
            regulariser_parameters: the parameters of the regulariser GP

        """
        self._regulariser_parameters = regulariser_parameters
        self._jit_compiled_calculate_regularisation = jax.jit(
            lambda parameters, x: self._calculate_regularisation(
                parameters=parameters,
                x=x,
            )
        )

    @abstractmethod
    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters,
        x: jnp.ndarray,
    ) -> jnp.float64:
        raise NotImplementedError

    def _calculate_regulariser_gaussian(
        self,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> Gaussian:
        """
        Calculates the regulariser Gaussian. This is either the prior or the Bayesian posterior of the regulariser GP
        conditioned on the inducing points.
        Args:
            x: the input data to calculate the regulariser Gaussian at
            full_covariance: whether to calculate the full covariance matrix or just the diagonal

        Returns: the regulariser Gaussian evaluated at the input data

        """
        if self._mode == RegularisationMode.prior:
            mean_p, covariance_q = self.regulariser.calculate_prior(
                parameters=self.regulariser_parameters,
                x=x,
                full_covariance=full_covariance,
            )
            return Gaussian(
                mean=mean_p,
                covariance=covariance_q,
            )
        elif self._mode == RegularisationMode.posterior:
            return self.regulariser.calculate_prediction_gaussian(
                parameters=self.regulariser_parameters,
                x=x,
                full_covariance=full_covariance,
            )

    def _calculate_regulariser_covariance(
        self,
        x: jnp.ndarray,
        full_covariance: bool,
    ) -> jnp.ndarray:
        """
        Calculates the regulariser covariance matrix. This is either the prior or the Bayesian posterior of the
        regulariser GP conditioned on the inducing points.
        Args:
            x: the input data to calculate the regulariser covariance matrix at
            full_covariance: whether to calculate the full covariance matrix or just the diagonal

        Returns:

        """
        if self._mode == RegularisationMode.prior:
            return self.regulariser.calculate_prior_covariance(
                parameters=self.regulariser_parameters,
                x=x,
                full_covariance=full_covariance,
            )
        elif self._mode == RegularisationMode.posterior:
            return self.regulariser.calculate_prediction_gaussian_covariance(
                parameters=self.regulariser_parameters,
                x=x,
                full_covariance=full_covariance,
            )

    @pydantic.validate_arguments(config=PYDANTIC_VALIDATION_CONFIG)
    def calculate_regularisation(
        self,
        parameters: Union[Dict, FrozenDict, GPBaseParameters],
        x: jnp.ndarray,
    ) -> jnp.float64:
        """
        Calculates the regularisation term.
        This calls the jitted function to calculate the regularisation term.
        Args:
            parameters: the parameters of the GP to regularise
            x: the input data to calculate the regularisation term at

        Returns: the regularisation term

        """
        if not isinstance(parameters, self.gp.Parameters):
            parameters = self.gp.generate_parameters(parameters)
        return self._jit_compiled_calculate_regularisation(
            parameters.dict(),
            x,
        )
