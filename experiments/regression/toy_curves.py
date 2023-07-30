from abc import ABC, abstractmethod

from jax import numpy as jnp
from jax import random

from src.utils.custom_types import PRNGKey


class Curve(ABC):
    @abstractmethod
    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        raise NotImplementedError


class Curve1(Curve):
    __name__ = "$y=2\sin(\pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(x * jnp.pi) + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve2(Curve):
    __name__ = "$y=1.2 \cos(2 \pi x)$ + x^2"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            1.2 * jnp.cos(x * (2 * jnp.pi))
            + x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve3(Curve):
    __name__ = "$y=\sin(1.5\pi x) + 0.3 \cos(4.5 \pi x) + 0.5 \sin(3.5 \pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(x * (1.5 * jnp.pi))
            + 0.3 * jnp.cos(x * (4.5 * jnp.pi))
            + 0.5 * jnp.sin(x * (3.5 * jnp.pi))
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve4(Curve):
    __name__ = "$y=2 \sin(\pi x) + x$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(jnp.pi * x) + x + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve5(Curve):
    __name__ = "$y=\sin(\pi x) + 0.3x^3$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(jnp.pi * x)
            + 0.3 * (x**3)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve6(Curve):
    __name__ = "$y=2\sin(\pi x) + \sin(3 \pi x) -x$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(x * jnp.pi)
            + jnp.sin(x * (3 * jnp.pi))
            - x
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve7(Curve):
    __name__ = "$y=2\cos(\pi x) + \sin(3 \pi x) -x^2$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.cos(x * jnp.pi)
            + jnp.sin(x * (3 * jnp.pi))
            - x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve8(Curve):
    __name__ = "$y=\sin(0.5 \pi (x-2)^2)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            jnp.sin(((x - 2) ** 2) * 0.5 * jnp.pi)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve9(Curve):
    __name__ = "$y=3\sqrt{4-x^2} + \sin(\pi x)$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            3 * jnp.sqrt(4 - x**2)
            + jnp.sin(jnp.pi * x)
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


class Curve10(Curve):
    __name__ = "$y=2 \sin(0.35 \pi (x-3)^2) + x^2$"

    def __call__(self, key: PRNGKey, x: jnp.ndarray, sigma_true: float) -> jnp.ndarray:
        return (
            2 * jnp.sin(((x - 3) ** 2) * 0.35 * jnp.pi)
            + x**2
            + sigma_true * random.normal(key, shape=x.shape)
        ).reshape(-1)


CURVE_FUNCTIONS = [
    Curve1(),
    Curve2(),
    Curve3(),
    Curve4(),
    Curve5(),
    Curve6(),
    Curve7(),
    Curve8(),
    Curve9(),
    Curve10(),
]


if __name__ == "__main__":
    import os

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from neural_tangents import stax

    from experiments.data import set_up_experiment
    from experiments.neural_networks import MultiLayerPerceptron
    from experiments.regression.plotters import plot_regression
    from src.kernels import CustomKernel
    from src.utils.custom_types import PRNGKey

    jax.config.update("jax_enable_x64", True)
    NUMBER_OF_DATA_POINTS = 500
    SIGMA_TRUE = 0.5
    TRAIN_DATA_PERCENTAGE = 0.8
    NUMBER_OF_TEST_INTERVALS = 2
    TOTAL_NUMBER_OF_INTERVALS = 8
    NUMBER_OF_INDUCING_POINTS = int(np.sqrt(NUMBER_OF_DATA_POINTS))
    REFERENCE_GP_LR = 1e-3
    REFERENCE_GP_TRAINING_EPOCHS = 5000
    REFERENCE_SAVE_CHECKPOINT_FREQUENCY = 1000
    REFERENCE_GP_BATCH_SIZE = 100
    REFERENCE_LOAD_CHECKPOINT = False
    OUTPUT_DIRECTORY = "outputs"
    EL_MATRIX_LOWER_BOUND = 1e-10
    INCLUDE_EIGENDECOMPOSITION = False
    APPROXIMATE_GP_LR = 1e-4
    APPROXIMATE_GP_TRAINING_EPOCHS = 200000
    APPROXIMATE_SAVE_CHECKPOINT_FREQUENCY = 1000
    APPROXIMATE_GP_BATCH_SIZE = 500
    APPROXIMATE_LOAD_CHECKPOINT = False
    TEMPERED_GP_LR = 1e-3
    TEMPERED_GP_TRAINING_EPOCHS = 2000
    TEMPERED_SAVE_CHECKPOINT_FREQUENCY = 1000
    TEMPERED_GP_BATCH_SIZE = 500
    TEMPERED_LOAD_CHECKPOINT = False
    X = jnp.linspace(-2, 2, NUMBER_OF_DATA_POINTS, dtype=np.float64).reshape(-1, 1)

    _, _, kernel_fn = stax.serial(
        stax.Dense(10, W_std=5, b_std=5),
        stax.Erf(),
        stax.Dense(1, W_std=5, b_std=5),
    )
    KERNEL = CustomKernel(lambda x1, x2: kernel_fn(x1, x2, "nngp"))
    KERNEL_PARAMETERS = KERNEL.Parameters()
    NEURAL_NETWORK = MultiLayerPerceptron([1, 10, 1])

    for i, CURVE_FUNCTION in enumerate(CURVE_FUNCTIONS):
        np.random.seed(i)
        KEY, SUBKEY = jax.random.split(jax.random.PRNGKey(i))
        curve_name = type(CURVE_FUNCTION).__name__.lower()
        output_folder = os.path.join(OUTPUT_DIRECTORY, curve_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        KEY, SUBKEY = jax.random.split(KEY)
        experiment_data = set_up_experiment(
            key=SUBKEY,
            curve_function=CURVE_FUNCTION,
            x=X,
            sigma_true=SIGMA_TRUE,
            number_of_test_intervals=NUMBER_OF_TEST_INTERVALS,
            total_number_of_intervals=TOTAL_NUMBER_OF_INTERVALS,
            number_of_inducing_points=NUMBER_OF_INDUCING_POINTS,
            train_data_percentage=TRAIN_DATA_PERCENTAGE,
            kernel=KERNEL,
            kernel_parameters=KERNEL_PARAMETERS,
        )
        fig = plot_regression(
            experiment_data=experiment_data,
            title=f"{CURVE_FUNCTION.__name__}",
        )
        fig.savefig(os.path.join(output_folder, f"{curve_name}.png"))
        plt.close(fig)
