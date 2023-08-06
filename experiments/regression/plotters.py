import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.data import Data, ExperimentData


def plot_regression(
    experiment_data: ExperimentData,
    inducing_data: Data = None,
    title: str = None,
    mean: jnp.ndarray = None,
    covariance: jnp.ndarray = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.tight_layout()
    if mean is not None and covariance is not None:
        ax.plot(experiment_data.full.x.reshape(-1), mean.reshape(-1), label="mean")
        stdev = jnp.sqrt(covariance)
        ax.fill_between(
            experiment_data.full.x.reshape(-1),
            (mean - 1.96 * stdev).reshape(-1),
            (mean + 1.96 * stdev).reshape(-1),
            facecolor=(0.8, 0.8, 0.8),
            label="error bound (95%)",
        )
    if title is not None:
        ax.set_title(title)
    ax.scatter(
        experiment_data.train.x,
        experiment_data.train.y,
        label="train",
        alpha=0.3,
        color="tab:blue",
    )
    ax.scatter(
        experiment_data.validation.x,
        experiment_data.validation.y,
        label="validation",
        alpha=0.3,
        color="tab:green",
    )
    ax.scatter(
        experiment_data.test.x,
        experiment_data.test.y,
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
    return fig
