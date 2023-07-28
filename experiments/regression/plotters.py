from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt

from experiments.data import ExperimentData


def plot_regression(
    experiment_data: ExperimentData,
    title: str = None,
    mean: jnp.ndarray = None,
    covariance: jnp.ndarray = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    if mean is not None and covariance is not None:
        ax.plot(experiment_data.x.reshape(-1), mean.reshape(-1), label="mean")
        stdev = jnp.sqrt(covariance)
        ax.fill_between(
            experiment_data.x.reshape(-1),
            (mean - 1.96 * stdev).reshape(-1),
            (mean + 1.96 * stdev).reshape(-1),
            facecolor=(0.8, 0.8, 0.8),
            label="error bound (95%)",
        )
    if title is not None:
        ax.set_title(title)
    ax.scatter(
        experiment_data.x_train,
        experiment_data.y_train,
        label="train",
        alpha=0.3,
        color="tab:blue",
    )
    ax.scatter(
        experiment_data.x_validation,
        experiment_data.y_validation,
        label="validation",
        alpha=0.3,
        color="tab:green",
    )
    ax.scatter(
        experiment_data.x_test,
        experiment_data.y_test,
        label="test",
        alpha=0.3,
        color="tab:orange",
    )
    ax.scatter(
        experiment_data.x_inducing,
        experiment_data.y_inducing,
        label="inducing",
        color="black",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig


def plot_losses(losses: List[float], loss_name: str, title: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_name)
    if title is not None:
        ax.set_title(title)
    return fig


def plot_two_losses(
    loss1: List[float],
    loss1_name: str,
    loss2: List[float],
    loss2_name: str,
    title: str = None,
):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    ax1.plot(loss1, "g-", label=loss1_name)
    ax2.plot(loss2, "b-", label=loss2_name)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss1_name, color="g")
    ax2.set_ylabel(loss2_name, color="b")
    if title is not None:
        ax1.set_title(title)
    return fig
