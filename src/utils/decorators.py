from functools import wraps
from typing import Any, Dict

import jax.numpy as jnp

from src.parameters.module import ModuleParameters


def preprocess_inputs(f):
    @wraps(f)
    def decorated_f(
        self,
        parameters: ModuleParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Preprocesses the inputs to the kernel.
        If x or y is 1D, it is converted to a 2D array with a single feature.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            self: the module
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: a dictionary containing the parameters, x, and y

        """
        # add dimension when x is 1D, assume the vector is a single feature of shape (1, n)
        x = jnp.atleast_2d(x)
        if y is not None:
            y = jnp.atleast_2d(y)
        return f(self, parameters, x, y, *args, **kwargs)

    return decorated_f


def check_inputs(f):
    @wraps(f)
    def decorated_f(
        self,
        parameters: ModuleParameters,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Checks that the kernel input dimensions are correct.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            self: the module
            parameters: parameters of the kernel
            x: design matrix of shape (n, d)
            y: design matrix of shape (m, d)

        Returns: a dictionary containing the parameters, x, and y

        """
        if y is not None:
            # check that the number of dimensions match
            assert (
                x.shape[1] == y.shape[1]
            ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"

            # check that x are matrices and not tensors
            if x.ndim != 2:
                raise ValueError(f"x cannot have more than two dimensions, {x.ndim=}")

            # check that y are matrices and not tensors
            if y.ndim != 2:
                raise ValueError(f"y cannot have more than two dimensions, {y.ndim=}")

        return f(self, parameters, x, y, *args, **kwargs)

    return decorated_f
