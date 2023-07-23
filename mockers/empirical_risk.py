import jax.numpy as jnp

from mockers.gp import MockGP
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase, GPBaseParameters


class MockEmpiricalRisk(EmpiricalRiskBase):
    def __init__(self, mock_empirical_risk: float, gp: GPBase = None):
        if gp is None:
            gp = MockGP()
        super().__init__(gp)
        self.mock_empirical_risk = mock_empirical_risk

    def _calculate_empirical_risk(
        self,
        parameters: GPBaseParameters = None,
        x: jnp.ndarray = None,
        y: jnp.ndarray = None,
    ) -> jnp.float64:
        return self.mock_empirical_risk
