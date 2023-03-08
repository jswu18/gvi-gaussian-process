# Copied from
# https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434

from typing import Generic, TypeVar

import jax.numpy as jnp
from pydantic.fields import ModelField

JSON_ENCODERS = {jnp.ndarray: lambda arr: arr.tolist()}

DType = TypeVar("DType")


class ArrayType(jnp.ndarray, Generic[DType]):
    """
    Wrapper class for jax.numpy arrays that stores and validates type information.
    This can be used in place of a numpy array, but when used in a pydantic BaseModel
    or with pydantic.validate_arguments, its dtype will be *coerced* at runtime to the
    declared type.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val, field: ModelField):
        dtype_field = field.sub_fields[0]
        actual_dtype = dtype_field.type_.__args__[0]
        # If numpy cannot create an array with the request dtype, an error will be raised
        # and correctly bubbled up.
        array = jnp.array(val, dtype=actual_dtype)
        return array
