from experiments.schemes import EmpiricalRiskScheme, RegularisationScheme
from src.empirical_risks import NegativeLogLikelihood
from src.empirical_risks.base import EmpiricalRiskBase
from src.gps.base.base import GPBase, GPBaseParameters
from src.regularisations import (
    SquaredDifferenceRegularisation,
    WassersteinRegularisation,
)
from src.regularisations.base import RegularisationBase
from src.regularisations.point_wise import (
    PointWiseBhattacharyyaRegularisation,
    PointWiseHellingerRegularisation,
    PointWiseKLRegularisation,
    PointWiseRenyiRegularisation,
    PointWiseSymmetricKLRegularisation,
    PointWiseWassersteinRegularisation,
)


def resolve_empirical_risk(
    empirical_risk_scheme: EmpiricalRiskScheme, gp: GPBase
) -> EmpiricalRiskBase:
    if empirical_risk_scheme == EmpiricalRiskScheme.negative_log_likelihood:
        return NegativeLogLikelihood(gp=gp)
    else:
        raise ValueError(f"Unknown empirical risk scheme: {empirical_risk_scheme=}")


def resolve_regularisation(
    regularisation_scheme: RegularisationScheme,
    gp: GPBase,
    regulariser: GPBase,
    regulariser_parameters: GPBaseParameters,
) -> RegularisationBase:
    if regularisation_scheme == RegularisationScheme.squared_difference:
        return SquaredDifferenceRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.wasserstein:
        return WassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_wasserstein:
        return PointWiseWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_kl:
        return PointWiseKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_symmetric_kl:
        return PointWiseSymmetricKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_bhattacaryya:
        return PointWiseBhattacharyyaRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_hellinger:
        return PointWiseHellingerRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == RegularisationScheme.point_wise_renyi:
        return PointWiseRenyiRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    else:
        raise ValueError(f"Unknown regularisation scheme: {regularisation_scheme=}")
