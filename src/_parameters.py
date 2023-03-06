from __future__ import annotations

import operator
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from typing import Callable


def _check_parameter_type(func: Callable) -> Callable:
    """Makes sure parameters match type before performing an operation.

    Args:
        func: an operation function

    Returns:
        The same operation function, wrapped with a typing check assertion.
    """

    @wraps(func)
    def wrapped(self, parameters: Parameters) -> Parameters:
        assert type(self) == type(
            parameters
        ), f"Operation requires same parameter type {type(self)=} != {type(parameters)=}"
        return func(self, parameters)

    return wrapped


@dataclass
class Parameters(ABC):
    """
    An abstract dataclass which defines and keeps track of parameters required for different kernels, models, etc.
    """

    def _perform_operation(
        self, parameters: Parameters, operation: operator
    ) -> Parameters:
        """Performs an operation between parameter dataclasses.

        Args:
            parameters: parameter object to apply operation to current parameter object
            operation: operation to perform

        Returns:
            A new parameter dataclass containing the operation between the current and given parameter object
        """
        update = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            update_value = getattr(parameters, field)
            if isinstance(update_value, dict):
                update_value = value.__class__(**update_value)
            update[field] = operation(value, update_value)
        return self.__class__(**update)

    @_check_parameter_type
    def __add__(self, parameters: Parameters) -> Parameters:
        """Perform a summation of parameter dataclasses.

        Args:
            parameters: parameter object to add to the current parameter object

        Returns:
            A parameter dataclass adding the parameters of the current and given parameter object.
        """
        return self._perform_operation(parameters, operation=operator.add)

    @_check_parameter_type
    def __sub__(self, parameters: Parameters) -> Parameters:
        """Perform a summation of parameter dataclasses.

        Args:
            parameters: parameter object to subtract from the current parameter object

        Returns:
            A parameter dataclass subtracting the given parameter dataclass from the current parameter dataclass
        """
        return self._perform_operation(parameters, operation=operator.sub)
