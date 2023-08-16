from experiments.shared import schemas
from src.gps.base.base import GPBase, GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.regularisations import (
    GaussianSquaredDifferenceRegularisation,
    GaussianWassersteinRegularisation,
    MultinomialWassersteinRegularisation,
)
from src.regularisations.base import RegularisationBase
from src.regularisations.point_wise import (
    PointWiseBhattacharyyaRegularisation,
    PointWiseGaussianWassersteinRegularisation,
    PointWiseHellingerRegularisation,
    PointWiseKLRegularisation,
    PointWiseRenyiRegularisation,
)


def regularisation_resolver(
    regularisation_schema: schemas.RegularisationSchema,
    gp: GPBase,
    regulariser: GPBase,
    regulariser_parameters: GPBaseParameters,
) -> RegularisationBase:
    if (
        regularisation_schema
        == schemas.RegularisationSchema.gaussian_squared_difference
    ):
        return GaussianSquaredDifferenceRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.gaussian_wasserstein:
        return GaussianWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif (
        regularisation_schema
        == schemas.RegularisationSchema.point_wise_gaussian_wasserstein
    ):
        return PointWiseGaussianWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.point_wise_kl:
        return PointWiseKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.point_wise_bhattacharyya:
        return PointWiseBhattacharyyaRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.point_wise_hellinger:
        return PointWiseHellingerRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.point_wise_renyi:
        return PointWiseRenyiRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    elif regularisation_schema == schemas.RegularisationSchema.multinomial_wasserstein:
        assert isinstance(gp, GPClassificationBase), "GP must be a classification GP"
        assert isinstance(
            regulariser, GPClassificationBase
        ), "Regulariser must be a classification GP"
        return MultinomialWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
        )
    else:
        raise ValueError(f"Unknown regularisation schema: {regularisation_schema=}")
