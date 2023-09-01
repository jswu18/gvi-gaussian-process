import jax.numpy as jnp
import pytest
from jax import random
from jax.config import config

from mockers.kernel import MockKernel, MockKernelParameters
from src.inducing_points_selection import (
    ConditionalVarianceInducingPointsSelector,
    InducingPointsSelectorBase,
    RandomInducingPointsSelector,
)

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "x,x_inducing,inducing_points_selector",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                    [1.6, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [4.5, 2.5, 3.5],
                    [1.5, 2.5, 3.5],
                ]
            ),
            ConditionalVarianceInducingPointsSelector(),
        ],
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                    [1.6, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                ]
            ),
            RandomInducingPointsSelector(),
        ],
    ],
)
def test_inducing_points_selection(
    x: jnp.ndarray,
    x_inducing: jnp.ndarray,
    inducing_points_selector: InducingPointsSelectorBase,
):
    x_inducing_selected, _ = inducing_points_selector.compute_inducing_points(
        key=random.PRNGKey(0),
        training_inputs=x,
        number_of_inducing_points=2,
        kernel=MockKernel(),
        kernel_parameters=MockKernelParameters(),
    )
    assert jnp.array_equal(
        x_inducing_selected,
        x_inducing,
    )


@pytest.mark.parametrize(
    "x,x_inducing,threshold",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                    [1.6, 2.5, 3.5],
                ]
            ),
            jnp.array(
                [
                    [4.5, 2.5, 3.5],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0.0,
        ],
    ],
)
def test_inducing_points_selection(
    x: jnp.ndarray,
    x_inducing: jnp.ndarray,
    threshold: float,
):
    inducing_points_selector = ConditionalVarianceInducingPointsSelector(
        threshold=threshold,
    )
    x_inducing_selected, _ = inducing_points_selector.compute_inducing_points(
        key=random.PRNGKey(0),
        training_inputs=x,
        number_of_inducing_points=2,
        kernel=MockKernel(),
        kernel_parameters=MockKernelParameters(),
    )
    assert jnp.array_equal(
        x_inducing_selected,
        x_inducing,
    )


@pytest.mark.parametrize(
    "x,x_inducing_indices,inducing_points_selector",
    [
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                    [1.6, 2.5, 3.5],
                ]
            ),
            jnp.array([2, 1]),
            ConditionalVarianceInducingPointsSelector(),
        ],
        [
            jnp.stack(
                [
                    2 * jnp.ones((28, 28, 1)),
                    4 * jnp.ones((28, 28, 1)),
                    5 * jnp.ones((28, 28, 1)),
                ]
            ),
            jnp.array([2, 1]),
            ConditionalVarianceInducingPointsSelector(),
        ],
        [
            jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [4.5, 2.5, 3.5],
                    [1.6, 2.5, 3.5],
                ]
            ),
            jnp.array([1, 2]),
            RandomInducingPointsSelector(),
        ],
        [
            jnp.stack(
                [
                    2 * jnp.ones((28, 28, 1)),
                    4 * jnp.ones((28, 28, 1)),
                    5 * jnp.ones((28, 28, 1)),
                ]
            ),
            jnp.array([0, 2]),
            RandomInducingPointsSelector(),
        ],
    ],
)
def test_inducing_points_indices_selection(
    x: jnp.ndarray,
    x_inducing_indices: jnp.ndarray,
    inducing_points_selector: InducingPointsSelectorBase,
):
    _, x_inducing_selected_indices = inducing_points_selector.compute_inducing_points(
        key=random.PRNGKey(0),
        training_inputs=x,
        number_of_inducing_points=2,
        kernel=MockKernel(),
        kernel_parameters=MockKernelParameters(),
    )
    assert jnp.array_equal(
        x_inducing_selected_indices,
        x_inducing_indices,
    )
