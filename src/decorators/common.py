from functools import wraps
from typing import Any, Dict

from flax.core import FrozenDict
from flax.core.frozen_dict import unfreeze
from jax import numpy as jnp


def default_duplicate_x(f):
    @wraps(f)
    def decorated_f(
        self,
        parameters: FrozenDict,
        x: jnp.ndarray,
        y: jnp.ndarray = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Default behaviour for missing inputs.
        If y is None, it is set to x.
            - n is the number of points in x
            - m is the number of points in y
            - d is the number of dimensions
        Args:
            self: the module
            parameters: parameters of the module
            x: matrix of shape (n, d)
            y: matrix of shape (m, d)

        Returns: a dictionary containing the parameters, x, and y

        """
        if y is None:
            y = x
        return f(self, parameters, x, y, *args, **kwargs)

    return decorated_f


def check_parameters(parameter_keys: Dict[str, type]):
    def _check_parameters(f):
        @wraps(f)
        def decorated_f(self, parameters, *args, **kwargs) -> FrozenDict:
            for key in parameters:
                if key not in parameters:
                    raise KeyError(f"{key} is not in the provided parameters")
                if not isinstance(parameters[key], parameter_keys[key]):
                    try:
                        parameters = unfreeze(parameters)
                        parameters[key] = parameter_keys[key](parameters[key])
                    except TypeError:
                        TypeError(
                            f"{key} needs type {parameter_keys[key]} not {type(parameters[key])=}"
                        )
            return f(self, parameters, *args, **kwargs)

        return decorated_f

    return _check_parameters
