from typing import List, Union

from matplotlib import pyplot as plt


def plot_losses(
    losses: Union[List[float], List[List[float]]],
    loss_name: str,
    save_path: str,
    labels: Union[str, List[str]] = None,
    title: str = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.tight_layout()
    if isinstance(losses[0], list):
        for loss, label in zip(losses, labels):
            ax.plot(loss, label=label)
    else:
        ax.plot(losses, label=labels)
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_name)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_two_losses(
    loss1: List[float],
    loss1_name: str,
    loss2: List[float],
    loss2_name: str,
    save_path: str,
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
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
