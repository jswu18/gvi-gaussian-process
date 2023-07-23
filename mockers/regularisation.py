import jax.numpy as jnp

from mockers.gp import MockGP, MockGPParameters
from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations.base import RegularisationBase


class MockRegularisation(RegularisationBase):
    def __init__(
        self,
        mock_regularisation: float,
        gp: GPBase = None,
        regulariser: GPBase = None,
        regulariser_parameters: GPBaseParameters = None,
    ):
        if gp is None:
            gp = MockGP()
        if regulariser is None:
            regulariser = MockGP()
        if regulariser_parameters is None:
            regulariser_parameters = MockGPParameters()

        super().__init__(gp, regulariser, regulariser_parameters)
        self.mock_regularisation = mock_regularisation

    def _calculate_regularisation(
        self,
        parameters: GPBaseParameters = None,
        x: jnp.ndarray = None,
    ) -> jnp.float64:
        return self.mock_regularisation
