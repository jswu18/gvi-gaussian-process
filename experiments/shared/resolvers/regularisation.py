from typing import Dict, Union

from flax.core.frozen_dict import FrozenDict

from experiments.shared import schemas
from src.gps.base.base import GPBase, GPBaseParameters
from src.gps.base.classification_base import GPClassificationBase
from src.regularisations import (
    GaussianSquaredDifferenceRegularisation,
    GaussianWassersteinRegularisation,
    MultinomialWassersteinRegularisation,
)
from src.regularisations.base import RegularisationBase
from src.regularisations.projected import (
    ProjectedBhattacharyyaRegularisation,
    ProjectedGaussianWassersteinRegularisation,
    ProjectedHellingerRegularisation,
    ProjectedKLRegularisation,
    ProjectedRenyiRegularisation,
)


def regularisation_resolver(
    regularisation_config: Union[FrozenDict, Dict],
    gp: GPBase,
    regulariser: GPBase,
    regulariser_parameters: GPBaseParameters,
) -> RegularisationBase:
    assert (
        "regularisation_schema" in regularisation_config
    ), "Regularisation schema must be specified"
    assert (
        "regularisation_kwargs" in regularisation_config
    ), "Regularisation kwargs must be specified"
    regularisation_schema = regularisation_config["regularisation_schema"]
    regularisation_kwargs = regularisation_config["regularisation_kwargs"]
    if (
        regularisation_schema
        == schemas.RegularisationSchema.gaussian_squared_difference
    ):
        return GaussianSquaredDifferenceRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif regularisation_schema == schemas.RegularisationSchema.gaussian_wasserstein:
        return GaussianWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif (
        regularisation_schema
        == schemas.RegularisationSchema.projected_gaussian_wasserstein
    ):
        return ProjectedGaussianWassersteinRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif regularisation_schema == schemas.RegularisationSchema.projected_kl:
        return ProjectedKLRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif regularisation_schema == schemas.RegularisationSchema.projected_bhattacharyya:
        return ProjectedBhattacharyyaRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif regularisation_schema == schemas.RegularisationSchema.projected_hellinger:
        return ProjectedHellingerRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
        )
    elif regularisation_schema == schemas.RegularisationSchema.projected_renyi:
        return ProjectedRenyiRegularisation(
            gp=gp,
            regulariser=regulariser,
            regulariser_parameters=regulariser_parameters,
            **regularisation_kwargs,
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
            **regularisation_kwargs,
        )
    else:
        raise ValueError(f"Unknown regularisation schema: {regularisation_schema=}")
