from typing import List

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_regression(
    x_inducing: jnp.ndarray = None,
    y_inducing: jnp.ndarray = None,
    x_train: jnp.ndarray = None,
    y_train: jnp.ndarray = None,
    x_test: jnp.ndarray = None,
    y_test: jnp.ndarray = None,
    x_validation: jnp.ndarray = None,
    y_validation: jnp.ndarray = None,
    title: str = None,
    x: jnp.ndarray = None,
    mean: jnp.ndarray = None,
    covariance: jnp.ndarray = None,
):
    fig, ax = plt.subplots(figsize=(20, 10))

    if x is not None and mean is not None and covariance is not None:
        ax.plot(x.reshape(-1), mean.reshape(-1), label="mean")
        stdev = jnp.sqrt(covariance)
        ax.fill_between(
            x.reshape(-1),
            (mean - 1.96 * stdev).reshape(-1),
            (mean + 1.96 * stdev).reshape(-1),
            facecolor=(0.8, 0.8, 0.8),
            label="error bound (95%)",
        )
    if title is not None:
        ax.set_title(title)
    if x_train is not None and y_train is not None:
        ax.scatter(x_train, y_train, label="train", alpha=0.3, color="tab:blue")
    if x_validation is not None and y_validation is not None:
        ax.scatter(
            x_validation, y_validation, label="validation", alpha=0.3, color="tab:green"
        )
    if x_test is not None and y_test is not None:
        ax.scatter(x_test, y_test, label="test", alpha=0.3, color="tab:orange")
    if x_inducing is not None and y_inducing is not None:
        ax.scatter(x_inducing, y_inducing, label="inducing", color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig


def plot_losses(losses: List[float], loss_name: str, title: str = None):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_name)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return fig
