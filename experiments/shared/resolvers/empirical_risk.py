from experiments.shared import schemas
from src.empirical_risks import NegativeLogLikelihood
from src.empirical_risks.base import EmpiricalRiskBase
from src.empirical_risks.cross_entropy import CrossEntropy
from src.gps.base.base import GPBase
from src.gps.base.classification_base import GPClassificationBase


def empirical_risk_resolver(
    empirical_risk_schema: schemas.EmpiricalRiskSchema, gp: GPBase
) -> EmpiricalRiskBase:
    if empirical_risk_schema == schemas.EmpiricalRiskSchema.negative_log_likelihood:
        return NegativeLogLikelihood(gp=gp)
    if empirical_risk_schema == schemas.EmpiricalRiskSchema.cross_entropy:
        assert isinstance(
            gp, GPClassificationBase
        ), "CrossEntropy is only for classification"
        return CrossEntropy(gp=gp)
    raise ValueError(f"Unknown empirical risk schema: {empirical_risk_schema=}")
