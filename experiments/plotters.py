from typing import List

from matplotlib import pyplot as plt


def plot_losses(losses: List[float], loss_name: str, title: str = None):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.tight_layout()
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
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    fig.tight_layout()

    ax2 = ax1.twinx()
    ax1.plot(loss1, "g-", label=loss1_name)
    ax2.plot(loss2, "b-", label=loss2_name)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel(loss1_name, color="g")
    ax2.set_ylabel(loss2_name, color="b")
    if title is not None:
        ax1.set_title(title)
    return fig
