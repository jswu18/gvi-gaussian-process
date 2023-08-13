from typing import Callable, List

import matplotlib.pyplot as plt

from experiments.shared.data import Data


def plot_images(
    data_list: List[Data],
    reshape_function: Callable,
    save_path: str,
    max_images: int = 100,
):
    count_per_class = [data.x.shape[0] for data in data_list]
    images_to_show_per_class = int(min(max_images, max(count_per_class)))
    fig, ax = plt.subplots(
        nrows=len(data_list),
        ncols=images_to_show_per_class,
        figsize=(images_to_show_per_class, len(data_list)),
    )

    for i, data in enumerate(data_list):
        for j, image in enumerate(data.x[:images_to_show_per_class]):
            ax[i, j].imshow(reshape_function(image))
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
