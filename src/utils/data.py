import math
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.custom_types import PRNGKey


def generate_batch(
    key: PRNGKey,
    data: Union[Tuple[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Adapted from:
    https://github.com/huggingface/transformers/blob/286a18fa0080dd39bd373008d11d831fbb1a77f1/examples/flax/summarization/run_summarization_flax.py#L339
    and
    https://www.kaggle.com/code/aakashnain/building-models-in-jax-part1-stax


    Return batches of size `batch_size` from `x` and `y`.
    If `drop_last` is set to `False`, the final batch may be incomplete, and range in size from 1 to `batch_size`.
    Shuffle batches if `shuffle` is `True`.
    Args:
        key: PRNGKey
        data: jnp.ndarray or tuple of jnp.ndarrays each of shape (dataset_size, ...)
        batch_size: int, size of batch
        shuffle: bool, if True, shuffle data before batching
        drop_last: bool, if True, drop last incomplete batch

    Returns: batches of size `batch_size` from `x` and `y`
    """
    if isinstance(data, tuple):
        dataset_size = data[0].shape[0]
    else:
        dataset_size = data.shape[0]

    if shuffle:
        batch_idx = jax.random.permutation(key, dataset_size)
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(dataset_size)

    if drop_last:
        steps_per_epoch = math.floor(dataset_size / batch_size)
    else:
        steps_per_epoch = math.ceil(dataset_size / batch_size)
    # Ensure at least one step
    steps_per_epoch = max(steps_per_epoch, 1)

    for idx in range(steps_per_epoch):
        curr_idx = batch_idx[idx * batch_size : (idx + 1) * batch_size]
        if isinstance(data, tuple):
            yield (x[curr_idx, ...] for x in data)
        else:
            yield data[curr_idx, ...]
