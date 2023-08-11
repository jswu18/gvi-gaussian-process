from experiments.shared import schemes
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


def regularisation_resolver(
    regularisation_scheme: schemes.RegularisationScheme,
    gp: GPBase,
    regulariser: GPBase,
    regulariser_parameters: GPBaseParameters,
) -> RegularisationBase:
    if regularisation_scheme == schemes.RegularisationScheme.squared_difference:
        return SquaredDifferenceRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.wasserstein:
        return WassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_wasserstein:
        return PointWiseWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_kl:
        return PointWiseKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_symmetric_kl:
        return PointWiseSymmetricKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_bhattacharyya:
        return PointWiseBhattacharyyaRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_hellinger:
        return PointWiseHellingerRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_scheme == schemes.RegularisationScheme.point_wise_renyi:
        return PointWiseRenyiRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    else:
        raise ValueError(f"Unknown regularisation scheme: {regularisation_scheme=}")
