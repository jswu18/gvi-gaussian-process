import pdb

import pandas as pd
from jax import numpy as jnp
from jax import scipy as jsp

from experiments.shared.data import Data, ExperimentData
from experiments.shared.schemas import ActionSchema
from src.distributions import Gaussian
from src.gps.base.base import GPBaseParameters
from src.gps.base.regression_base import GPRegressionBase


def calculate_negative_log_likelihood(
    data: Data, y_std, gaussian: Gaussian
) -> pd.DataFrame:
    out = []
    for loc, variance, y in zip(
        gaussian.mean.reshape(-1), gaussian.covariance.reshape(-1), data.y.reshape(-1)
    ):
        out.append(
            -jsp.stats.norm.logpdf(
                y,
                loc=float(loc),
                scale=float(jnp.sqrt(variance)),
            )
        )
    return jnp.mean(jnp.array(out)) + jnp.log(y_std)


def calculate_metrics(
    experiment_data: ExperimentData,
    gp: GPRegressionBase,
    gp_parameters: GPBaseParameters,
    action: ActionSchema,
    experiment_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    metrics_dict = {}
    if experiment_data.test.y is not None:
        gaussian = Gaussian(
            **gp.predict_probability(gp_parameters, x=experiment_data.test.x).dict()
        )
        metrics_dict["test"] = {
            "negative-log-likelihood": calculate_negative_log_likelihood(
                data=experiment_data.test,
                y_std=experiment_data.y_std,
                gaussian=gaussian,
            ),
        }
    if experiment_data.train.y is not None:
        gaussian = Gaussian(
            **gp.predict_probability(gp_parameters, x=experiment_data.train.x).dict()
        )
        metrics_dict["train"] = {
            "negative-log-likelihood": calculate_negative_log_likelihood(
                data=experiment_data.train,
                y_std=experiment_data.y_std,
                gaussian=gaussian,
            ),
        }
    if experiment_data.validation is not None:
        gaussian = Gaussian(
            **gp.predict_probability(
                gp_parameters, x=experiment_data.validation.x
            ).dict()
        )
        metrics_dict["validation"] = {
            "negative-log-likelihood": calculate_negative_log_likelihood(
                data=experiment_data.validation,
                y_std=experiment_data.y_std,
                gaussian=gaussian,
            ),
        }
    if experiment_data.full is not None:
        gaussian = Gaussian(
            **gp.predict_probability(gp_parameters, x=experiment_data.full.x).dict()
        )
        metrics_dict["full"] = {
            "negative-log-likelihood": calculate_negative_log_likelihood(
                data=experiment_data.full,
                y_std=experiment_data.y_std,
                gaussian=gaussian,
            ),
        }
    df = pd.DataFrame(metrics_dict)
    if "test" not in df.columns:
        df["test"] = jnp.nan
    if "train" not in df.columns:
        df["train"] = jnp.nan
    if "validation" not in df.columns:
        df["validation"] = jnp.nan
    if "full" not in df.columns:
        df["full"] = jnp.nan
    df["action"] = action.value
    df["experiment_name"] = experiment_name
    df["dataset_name"] = dataset_name
    return df.rename_axis("metric").reset_index()
