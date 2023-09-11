import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

# enable 64 bit
from jax.config import config

from src.empirical_risks import NegativeLogLikelihood
from src.generalised_variational_inference import GeneralisedVariationalInference
from src.gps import ApproximateGPRegression, GPRegression
from src.inducing_points_selection import ConditionalVarianceInducingPointsSelector
from src.kernels import TemperedKernel
from src.kernels.approximate import SparsePosteriorKernel
from src.kernels.standard import ARDKernel
from src.means import ConstantMean, CustomMean
from src.regularisations.projected import ProjectedRenyiRegularisation

config.update("jax_enable_x64", True)


class FullyConnectedNeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=10,
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            features=1,
        )(x)
        return x


#######################################################################################################################
# GENERATE DATA
number_of_points = 100
noise = 0.1
key = jax.random.PRNGKey(0)

# generate data with noise
key, subkey = jax.random.split(key)
x = jnp.linspace(-1, 1, number_of_points).reshape(-1, 1)
y = jnp.sin(jnp.pi * x) + noise * jax.random.normal(subkey, shape=x.shape)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, label="train", alpha=0.3, color="tab:blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Train Data")
ax.legend()
fig.savefig("examples/train_data.png", bbox_inches="tight")

#######################################################################################################################
# CONSTRUCT KERNEL
kernel = ARDKernel(number_of_dimensions=1)
kernel_parameters = kernel.Parameters.construct(
    log_scaling=jnp.log(10.0), log_lengthscales=jnp.log(10.0)
)

#######################################################################################################################
# INDUCING POINT SELECTION
key, subkey = jax.random.split(key)
inducing_points_selector = ConditionalVarianceInducingPointsSelector()
(
    inducing_points,
    inducing_points_indices,
) = inducing_points_selector.compute_inducing_points(
    key=subkey,
    training_inputs=x,
    number_of_inducing_points=int(jnp.sqrt(number_of_points)),
    kernel=kernel,
    kernel_parameters=kernel_parameters,
)
inducing_points_responses = y[inducing_points_indices]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, label="train", alpha=0.3, color="tab:blue")
ax.scatter(inducing_points, inducing_points_responses, label="inducing", color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Inducing Points Selection")
ax.legend()
fig.savefig("examples/inducing_data.png", bbox_inches="tight")

#######################################################################################################################
# CONSTRUCT EXACT GP
# construct mean
mean = ConstantMean()
mean_parameters = mean.Parameters.construct(constant=0.0)

# construct exact GP
exact_gp = GPRegression(
    mean=mean,
    kernel=kernel,
    x=inducing_points,
    y=inducing_points_responses,
)
exact_gp_parameters = exact_gp.Parameters.construct(
    log_observation_noise=jnp.log(1.0),
    mean=mean_parameters,
    kernel=kernel_parameters,
)

#######################################################################################################################
# CONSTRUCT EMPIRICAL RISK
empirical_risk = NegativeLogLikelihood(gp=exact_gp)

#######################################################################################################################
# TRAIN EXACT GP
empirical_risk_loss = [
    empirical_risk.calculate_empirical_risk(
        exact_gp_parameters,
        inducing_points,
        inducing_points_responses,
    )
]

optimiser = optax.adabelief(learning_rate=1e-3)
opt_state = optimiser.init(exact_gp_parameters.dict())
for _ in range(1000):
    gradients = jax.grad(
        lambda exact_gp_parameters_dict: empirical_risk.calculate_empirical_risk(
            exact_gp_parameters_dict,
            inducing_points,
            inducing_points_responses,
        )
    )(exact_gp_parameters.dict())
    updates, opt_state = optimiser.update(gradients, opt_state)
    exact_gp_parameters = exact_gp_parameters.construct(
        **optax.apply_updates(exact_gp_parameters.dict(), updates)
    )
    empirical_risk_loss.append(
        empirical_risk.calculate_empirical_risk(
            exact_gp_parameters,
            inducing_points,
            inducing_points_responses,
        )
    )

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(empirical_risk_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("NLL")
ax.set_title("Exact GP NLL")
fig.savefig("examples/exact_gp_nll.png", bbox_inches="tight")

#######################################################################################################################
# PLOT EXACT GP PREDICTION
prediction = exact_gp.predict_probability(
    parameters=exact_gp_parameters,
    x=x,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, prediction.mean.reshape(-1), label="mean")
stdev = jnp.sqrt(prediction.covariance)
ax.fill_between(
    x.reshape(-1),
    (prediction.mean - 1.96 * stdev).reshape(-1),
    (prediction.mean + 1.96 * stdev).reshape(-1),
    facecolor=(0.8, 0.8, 0.8),
    label="error bound (95%)",
)
ax.scatter(x, y, label="train", alpha=0.3, color="tab:blue")
ax.scatter(inducing_points, inducing_points_responses, label="inducing", color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Exact GP")
ax.legend()
fig.savefig("examples/exact_gp.png", bbox_inches="tight")

#######################################################################################################################
# CONSTRUCT NEURAL NETWORK
fcnn = FullyConnectedNeuralNetwork()

# randomly initialise parameters
key, subkey = jax.random.split(key)
fcnn_parameters = fcnn.init(
    subkey,
    jnp.empty(1),
)

#######################################################################################################################
# CONSTRUCT APPROXIMATE MEAN
approximate_mean = CustomMean(
    mean_function=lambda parameters, x: fcnn.apply(parameters, x)
)
approximate_mean_parameters = approximate_mean.Parameters.construct(
    custom=fcnn_parameters
)

#######################################################################################################################
# CONSTRUCT APPROXIMATE KERNEL
approximate_kernel = SparsePosteriorKernel(
    base_kernel=kernel,
    inducing_points=inducing_points,
)
approximate_kernel_parameters = approximate_kernel.Parameters.construct(
    base_kernel=kernel_parameters
)

#######################################################################################################################
# CONSTRUCT APPROXIMATE GP
approximate_gp = ApproximateGPRegression(
    mean=approximate_mean,
    kernel=approximate_kernel,
)
approximate_gp_parameters = approximate_gp.Parameters.construct(
    mean=approximate_mean_parameters,
    kernel=approximate_kernel_parameters,
)

#######################################################################################################################
# CONSTRUCT GVI
gvi_empirical_risk = NegativeLogLikelihood(gp=approximate_gp)
gvi_regularisation = ProjectedRenyiRegularisation(
    gp=approximate_gp,
    regulariser=exact_gp,
    regulariser_parameters=exact_gp_parameters,
    mode="posterior",
    alpha=0.5,
)
gvi = GeneralisedVariationalInference(
    empirical_risk=gvi_empirical_risk,
    regularisation=gvi_regularisation,
)

#######################################################################################################################
# TRAIN APPROXIMATE GP
optimiser = optax.adabelief(learning_rate=1e-3)
opt_state = optimiser.init(approximate_gp_parameters.dict())
gvi_loss = [
    gvi.calculate_loss(
        approximate_gp_parameters,
        x,
        y,
    )
]
for _ in range(10000):
    gradients = jax.grad(
        lambda approximate_gp_parameters_dict: gvi.calculate_loss(
            approximate_gp_parameters_dict,
            x,
            y,
        )
    )(approximate_gp_parameters.dict())
    updates, opt_state = optimiser.update(gradients, opt_state)
    approximate_gp_parameters = approximate_gp_parameters.construct(
        **optax.apply_updates(approximate_gp_parameters.dict(), updates)
    )
    gvi_loss.append(
        gvi.calculate_loss(
            approximate_gp_parameters,
            x,
            y,
        )
    )
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(gvi_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("GVI Loss")
ax.set_title("Approximate GP GVI Loss")
fig.savefig("examples/approximate_gp_gvi_loss.png", bbox_inches="tight")

#######################################################################################################################
# PLOT APPROXIMATE GP PREDICTION
prediction = approximate_gp.predict_probability(
    parameters=approximate_gp_parameters,
    x=x,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, prediction.mean.reshape(-1), label="mean")
stdev = jnp.sqrt(prediction.covariance)
ax.fill_between(
    x.reshape(-1),
    (prediction.mean - 1.96 * stdev).reshape(-1),
    (prediction.mean + 1.96 * stdev).reshape(-1),
    facecolor=(0.8, 0.8, 0.8),
    label="error bound (95%)",
)
ax.scatter(x, y, label="train", alpha=0.3, color="tab:blue")
ax.scatter(inducing_points, inducing_points_responses, label="inducing", color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Approximate GP")
ax.legend()
fig.savefig("examples/approximate_gp.png", bbox_inches="tight")

#######################################################################################################################
# GENERATE TEMPERING DATA
key, subkey = jax.random.split(key)
x_temper = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y_temper = jnp.sin(jnp.pi * x_temper) + noise * jax.random.normal(
    subkey, shape=x_temper.shape
)

#######################################################################################################################
# CONSTRUCT TEMPER KERNEL
tempered_kernel = TemperedKernel(
    base_kernel=approximate_gp.kernel,
    base_kernel_parameters=approximate_kernel_parameters.construct(
        **approximate_gp_parameters.kernel
    ),
    number_output_dimensions=1,
)
tempered_kernel_parameters = tempered_kernel.Parameters.construct(
    log_tempering_factor=jnp.log(1.0)
)

#######################################################################################################################
# CONSTRUCT TEMPER GP
tempered_gp = ApproximateGPRegression(
    mean=approximate_gp.mean,
    kernel=tempered_kernel,
)
tempered_gp_parameters = tempered_gp.Parameters.construct(
    mean=approximate_gp_parameters.mean,
    kernel=tempered_kernel_parameters,
)

# TRAIN TEMPER GP
tempered_empirical_risk = NegativeLogLikelihood(gp=tempered_gp)


tempered_empirical_risk_loss = [
    tempered_empirical_risk.calculate_empirical_risk(
        tempered_gp_parameters,
        x_temper,
        y_temper,
    )
]
optimiser = optax.adabelief(learning_rate=1e-3)

# initalise optimiser with tempered kernel parmaeters
opt_state = optimiser.init(tempered_kernel_parameters.dict())
for _ in range(1000):
    gradients = jax.grad(
        lambda tempered_kernel_parameters_dict: tempered_empirical_risk.calculate_empirical_risk(
            tempered_gp.Parameters.construct(
                log_observation_noise=approximate_gp_parameters.log_observation_noise,
                mean=approximate_gp_parameters.mean,
                kernel=tempered_kernel_parameters_dict,
            ),
            x_temper,
            y_temper,
        )
    )(tempered_kernel_parameters.dict())
    updates, opt_state = optimiser.update(gradients, opt_state)
    tempered_kernel_parameters = tempered_kernel_parameters.construct(
        **optax.apply_updates(tempered_kernel_parameters.dict(), updates)
    )
    tempered_gp_parameters = tempered_gp.Parameters.construct(
        log_observation_noise=approximate_gp_parameters.log_observation_noise,
        mean=approximate_gp_parameters.mean,
        kernel=tempered_kernel_parameters,
    )
    tempered_empirical_risk_loss.append(
        tempered_empirical_risk.calculate_empirical_risk(
            tempered_gp_parameters,
            x_temper,
            y_temper,
        )
    )
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(tempered_empirical_risk_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("NLL")
ax.set_title("Tempered Approximate GP NLL")
fig.savefig("examples/tempered_approximate_gp_nll.png", bbox_inches="tight")

#######################################################################################################################
# PLOT TEMPERED GP PREDICTION
prediction = tempered_gp.predict_probability(
    parameters=tempered_gp_parameters,
    x=x,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, prediction.mean.reshape(-1), label="mean")
stdev = jnp.sqrt(prediction.covariance)
ax.fill_between(
    x.reshape(-1),
    (prediction.mean - 1.96 * stdev).reshape(-1),
    (prediction.mean + 1.96 * stdev).reshape(-1),
    facecolor=(0.8, 0.8, 0.8),
    label="error bound (95%)",
)
ax.scatter(x, y, label="train", alpha=0.3, color="tab:blue")
ax.scatter(
    x_temper,
    y_temper,
    label="validation",
    alpha=0.3,
    color="tab:green",
)
ax.scatter(inducing_points, inducing_points_responses, label="inducing", color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Tempered Approximate GP")
ax.legend()
fig.savefig("examples/tempered_gp.png", bbox_inches="tight")
