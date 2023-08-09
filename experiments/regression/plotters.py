import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.data import Data, ExperimentData
from src.distributions import Gaussian
from src.gps.base.base import GPBaseParameters
from src.gps.base.regression_base import GPRegressionBase


def plot_data(
    save_path: str,
    train_data: Data = None,
    test_data: Data = None,
    validation_data: Data = None,
    inducing_data: Data = None,
    title: str = None,
    prediction_x: jnp.ndarray = None,
    mean: jnp.ndarray = None,
    covariance: jnp.ndarray = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.tight_layout()
    if mean is not None and covariance is not None and prediction_x is not None:
        ax.plot(prediction_x.reshape(-1), mean.reshape(-1), label="mean")
        stdev = jnp.sqrt(covariance)
        ax.fill_between(
            prediction_x.reshape(-1),
            (mean - 1.96 * stdev).reshape(-1),
            (mean + 1.96 * stdev).reshape(-1),
            facecolor=(0.8, 0.8, 0.8),
            label="error bound (95%)",
        )
    if title is not None:
        ax.set_title(title)
    if train_data is not None:
        ax.scatter(
            train_data.x,
            train_data.y,
            label="train",
            alpha=0.3,
            color="tab:blue",
        )
    if validation_data is not None:
        ax.scatter(
            validation_data.x,
            validation_data.y,
            label="validation",
            alpha=0.3,
            color="tab:green",
        )
    if test_data is not None:
        ax.scatter(
            test_data.x,
            test_data.y,
            label="test",
            alpha=0.3,
            color="tab:orange",
        )
    if inducing_data is not None:
        ax.scatter(
            inducing_data.x,
            inducing_data.y,
            label="inducing",
            color="black",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction(
    experiment_data: ExperimentData,
    inducing_data: Data,
    gp: GPRegressionBase,
    gp_parameters: GPBaseParameters,
    save_path: str,
    title: str = None,
):
    gaussian_prediction = Gaussian(
        **gp.predict_probability(
            parameters=gp_parameters,
            x=experiment_data.full.x,
        ).dict()
    )
    plot_data(
        train_data=experiment_data.train,
        test_data=experiment_data.test,
        validation_data=experiment_data.validation,
        inducing_data=inducing_data,
        title=title,
        prediction_x=experiment_data.full.x,
        mean=gaussian_prediction.mean,
        covariance=gaussian_prediction.covariance,
        save_path=save_path,
    )
