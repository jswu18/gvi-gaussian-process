import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import random

from src.kernels.base import KernelBase, KernelBaseParameters
from src.utils.samplers import sample_discrete_unnormalised_distribution

PRNGKey = Any  # pylint: disable=invalid-name


class InducingPointsSelector(ABC):
    @abstractmethod
    def compute_inducing_points(
        self,
        key: PRNGKey,
        training_inputs: jnp.ndarray,
        number_of_inducing_points: int,
        kernel: KernelBase,
        kernel_parameters: KernelBaseParameters,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py
        """
        raise NotImplementedError


class ConditionalVarianceInducingPointsSelector(InducingPointsSelector):
    """
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py
    """

    def __init__(
        self,
        sample: Optional[bool] = False,
        threshold: Optional[int] = 0.0,
    ):
        """

        Args:
            sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
                    point with highest variance at each iteration
            threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx
                       has converged.
        """
        self.sample = sample
        self.threshold = threshold

    def compute_inducing_points(
        self,
        key: PRNGKey,
        training_inputs: jnp.ndarray,
        number_of_inducing_points: int,
        kernel: KernelBase,
        kernel_parameters: KernelBaseParameters,
        jitter: float = 1e-12,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Selects inducing points based on the variance of the GP at the training points.

        Docstring from original code:
            The version of this code without sampling follows the Greedy approximation to MAP for DPPs in
            @incollection{NIPS2018_7805,
                    title = {Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity},
                    author = {Chen, Laming and Zhang, Guoxin and Zhou, Eric},
                    booktitle = {Advances in Neural Information Processing Systems 31},
                    year = {2018},
                }
            and the initial code is based on the implementation of this algorithm (https://github.com/laming-chen/fast-map-dpp)
            It is equivalent to running a partial pivoted Cholesky decomposition on Kff (see Figure 2 in the below ref.),
            @article{fine2001efficient,
                    title={Efficient SVM training using low-rank kernel representations},
                    author={Fine, Shai and Scheinberg, Katya},
                    journal={Journal of Machine Learning Research},
                    year={2001}
                }

            Initializes based on variance of noiseless GP fit on inducing points currently in active set
            Complexity: O(NM) memory, O(NM^2) time

        Args:
            key: PRNGKey, random key
            training_inputs: [N,D1,...,DN] numpy array,
            number_of_inducing_points: int, number of points desired. If threshold is None actual number returned
                                        may be less than number_of_inducing_points
            kernel: Kernel object, kernel to use for computing variance
            kernel_parameters: KernelParameters object, parameters for kernel
            jitter: float, jitter to add to diagonal of kernel matrix

        Returns:
            inducing inputs (number_of_inducing_points, D1, ..., DN)
            indices (number_of_inducing_points,)

        """
        assert number_of_inducing_points > 1, "Must have at least 2 inducing points"
        number_of_training_points = training_inputs.shape[0]
        key, subkey = random.split(key)
        perm = random.permutation(
            subkey, number_of_training_points
        )  # permute entries so tie-breaking is random
        training_inputs = training_inputs[perm, ...]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = (
            np.zeros(number_of_inducing_points, dtype=int) + number_of_training_points
        )
        di = (
            kernel.calculate_gram(
                parameters=kernel_parameters,
                x1=training_inputs,
                x2=training_inputs,
                full_covariance=False,
            )
            + jitter
        )
        if self.sample:
            key, subkey = random.split(key)
            indices[0] = sample_discrete_unnormalised_distribution(subkey, di)
        else:
            indices[0] = jnp.argmax(di)  # select first point, add to index 0
        if number_of_inducing_points == 1:
            indices = indices.astype(int)
            inducing_points = training_inputs[indices]
            indices = perm[indices]
            return inducing_points, indices
        ci = np.zeros(
            (number_of_inducing_points - 1, number_of_training_points)
        )  # [number_of_inducing_points,number_of_training_points]
        for m in range(number_of_inducing_points - 1):
            j = int(indices[m])  # int
            new_inducing_points = training_inputs[j : j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            gram_matrix_raw = np.array(
                kernel.calculate_gram(
                    parameters=kernel_parameters,
                    x1=training_inputs,
                    x2=new_inducing_points,
                    full_covariance=True,
                )
            )
            gram_matrix = np.round(np.squeeze(gram_matrix_raw), 20)  # [N]
            gram_matrix[j] += jitter
            ei = (gram_matrix - jnp.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei**2
            except FloatingPointError:
                pass
            di = jnp.clip(di, 0, None)
            if self.sample:
                key, subkey = random.split(key)
                indices[m + 1] = sample_discrete_unnormalised_distribution(subkey, di)
            else:
                indices[m + 1] = jnp.argmax(di)  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if jnp.sum(jnp.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                warnings.warn(
                    "ConditionalVariance: Terminating selection of inducing points early."
                )
                break
        indices = indices.astype(int)
        inducing_points = training_inputs[indices]
        indices = perm[indices]
        return inducing_points, jnp.array(indices)
