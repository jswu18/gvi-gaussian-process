import jax.numpy as jnp


def check_matching_dimensions(
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> None:
    """
    Checks that the kernel input dimensions are correct.
        - n is the number of points in x
        - m is the number of points in y
        - d is the number of dimensions
    Args:
        x: design matrix of shape (n, d)
        y: design matrix of shape (m, d)

    Returns: None

    """
    assert (
        x.shape[1] == y.shape[1]
    ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"


def check_maximum_dimension(x: jnp.ndarray, maximum_dimensionality: int) -> None:
    """
    Checks that the vector has a maximum number of dimensions.
    I.e. we are limited to matrices of two dimensions for kernel evaluations in this project.
    Args:
        x: an array
        maximum_dimensionality: the maximum allowed number of dimensions in x

    Returns: None

    """
    assert (
        x.ndim >= maximum_dimensionality
    ), f"Array cannot have more than {maximum_dimensionality} dimensions, {x.ndim=}"
