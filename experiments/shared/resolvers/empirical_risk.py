from experiments.shared import schemes
from src.empirical_risks import NegativeLogLikelihood
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase


def empirical_risk_resolver(
    empirical_risk_scheme: schemes.EmpiricalRiskScheme, gp: GPBase
) -> EmpiricalRiskBase:
    if empirical_risk_scheme == schemes.EmpiricalRiskScheme.negative_log_likelihood:
        return NegativeLogLikelihood(gp=gp)
    else:
        raise ValueError(f"Unknown empirical risk scheme: {empirical_risk_scheme=}")
