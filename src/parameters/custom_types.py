# Modified for JAX types from
# https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434

from typing import Generic, TypeVar

import jax.numpy as jnp
from pydantic.fields import ModelField

JSON_ENCODERS = {jnp.ndarray: lambda arr: arr.tolist()}

ArrayDType = TypeVar("ArrayDType")
FloatDType = TypeVar("FloatDType")


class JaxArrayType(jnp.ndarray, Generic[ArrayDType]):
    """
    Wrapper class for jax.numpy arrays that stores and validates type information.
    This can be used in place of a jax.numpy array, but when used in a pydantic BaseModel
    or with pydantic.validate_arguments, its dtype will be *coerced* at runtime to the
    declared type.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val, field: ModelField) -> jnp.ndarray:
        dtype_field = field.sub_fields[0]
        actual_dtype = dtype_field.type_.__args__[0]
        # If jax.numpy cannot create an array with the request dtype, an error will be raised
        # and correctly bubbled up.
        return jnp.array(val, dtype=actual_dtype)


class JaxFloatType(jnp.float64, Generic[FloatDType]):
    """
    Wrapper class for jax.float64 that stores and validates type information.
    This can be used in place of a jax.float64, but when used in a pydantic BaseModel
    or with pydantic.validate_arguments, its dtype will be *coerced* at runtime to the
    declared type.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val) -> jnp.float64:
        return jnp.float64(val)
