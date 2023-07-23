import jax.numpy as jnp
import pytest

from mockers.empirical_risk import MockEmpiricalRisk
from mockers.gp import MockGPParameters
from mockers.regularisation import MockRegularisation
from src import GeneralisedVariationalInference


@pytest.mark.parametrize(
    "regularisation,empirical_risk,gvi_loss",
    [
        [
            2.3,
            1.2,
            3.5,
        ],
    ],
)
def test_gvi(
    regularisation: float,
    empirical_risk: float,
    gvi_loss: float,
):
    regularisation = MockRegularisation(mock_regularisation=regularisation)
    empirical_risk = MockEmpiricalRisk(mock_empirical_risk=empirical_risk)
    gvi = GeneralisedVariationalInference(
        regularisation=regularisation,
        empirical_risk=empirical_risk,
    )
    assert (
        gvi.calculate_loss(
            parameters=MockGPParameters(),
            x=jnp.ones((1, 1)),
            y=jnp.ones((1,)),
        )
        == gvi_loss
    )
