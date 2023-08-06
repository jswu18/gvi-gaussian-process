from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_images(
    x: jnp.ndarray,
    y: jnp.ndarray,
    reshape_function: Callable,
    max_images: int = 10,
):
    y = jnp.argmax(y, axis=1)  # convert one-hot to class labels
    unique_classes, count_per_class = jnp.unique(y, return_counts=True)
    images_to_show_per_class = int(min(max_images, max(count_per_class)))
    fig, ax = plt.subplots(
        nrows=len(unique_classes),
        ncols=images_to_show_per_class,
        figsize=(images_to_show_per_class, len(unique_classes)),
    )

    for i, class_label in enumerate(unique_classes):
        for j, image in enumerate(x[y == class_label][:images_to_show_per_class]):
            ax[i, j].imshow(reshape_function(image))
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.tight_layout()
    return fig
