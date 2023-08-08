import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax
from flax.core.frozen_dict import FrozenDict
from flax.training import orbax_utils
from tqdm import tqdm

from experiments.data import Data
from experiments.schemes import Optimiser
from src.gps.base.base import GPBase, GPBaseParameters
from src.module import ModuleParameters
from src.utils.data import generate_batch

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


@dataclass
class TrainerSettings:
    key: int
    optimiser_scheme: Optimiser
    learning_rate: float
    number_of_epochs: int
    batch_size: int
    batch_shuffle: bool
    batch_drop_last: bool


class Trainer:
    def __init__(
        self,
        save_checkpoint_frequency: int,
        checkpoint_path: str,
        post_epoch_callback: Callable[[ModuleParameters], Dict[str, float]],
        break_condition_function: Callable[[ModuleParameters], bool] = None,
    ):
        self.save_checkpoint_frequency = save_checkpoint_frequency
        self.checkpoint_path = checkpoint_path
        self.post_epoch_callback = post_epoch_callback
        self.break_condition_function = break_condition_function

    @staticmethod
    def resolve_optimiser(
        optimiser: Optimiser, learning_rate: float
    ) -> optax.GradientTransformation:
        if optimiser == Optimiser.adam:
            return optax.adam(learning_rate=learning_rate)
        elif optimiser == Optimiser.adabeleif:
            return optax.sgd(learning_rate=learning_rate)
        elif optimiser == Optimiser.rmsprop:
            return optax.rmsprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimiser: {optimiser=}")

    def train(
        self,
        trainer_settings: TrainerSettings,
        parameters: ModuleParameters,
        data: Data,
        loss_function: Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], float],
        disable_tqdm: bool = False,
    ) -> Tuple[ModuleParameters, List[Dict[str, float]]]:
        post_epoch_history = []
        optimiser = Trainer.resolve_optimiser(
            trainer_settings.optimiser_scheme, trainer_settings.learning_rate
        )
        opt_state = optimiser.init(parameters.dict())
        key = jax.random.PRNGKey(trainer_settings.key)
        for epoch in tqdm(
            range(trainer_settings.number_of_epochs), disable=disable_tqdm
        ):
            if epoch % self.save_checkpoint_frequency == 0:
                ckpt = parameters.dict()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    os.path.join(self.checkpoint_path, f"epoch-{epoch}.ckpt"),
                    ckpt,
                    save_args=save_args,
                    force=True,
                )
            key, subkey = jax.random.split(key)
            batch_generator = generate_batch(
                key=subkey,
                data=(data.x, data.y),
                batch_size=trainer_settings.batch_size,
                shuffle=trainer_settings.batch_shuffle,
                drop_last=trainer_settings.batch_drop_last,
            )
            data_batch = next(batch_generator, None)
            while data_batch is not None:
                x_batch, y_batch = data_batch
                gradients = jax.grad(
                    lambda parameters_dict: loss_function(
                        parameters_dict,
                        x_batch,
                        y_batch,
                    )
                )(parameters.dict())
                updates, opt_state = optimiser.update(gradients, opt_state)
                parameters = parameters.construct(
                    **optax.apply_updates(parameters.dict(), updates)
                )
                data_batch = next(batch_generator, None)
            post_epoch_history.append(self.post_epoch_callback(parameters))
            if self.break_condition_function and self.break_condition_function(
                parameters
            ):
                break
        return parameters, post_epoch_history


#
# def train_nll(
#     gp: GPBase,
#     gp_parameters: GPBaseParameters,
#     data: Data,
#     training_parameters: TrainingParameters,
#     nll_break_condition: float = -float("inf"),
# ) -> Tuple[GPBaseParameters, List[float]]:
#     losses = []
#     opt_state = training_parameters.optimiser_scheme.init(gp_parameters.dict())
#     nll_loss = NegativeLogLikelihood(gp=gp)
#     key = training_parameters.key
#     for epoch in tqdm(range(training_parameters.number_of_epochs)):
#         if epoch % training_parameters.save_checkpoint_frequency == 0:
#             ckpt = gp_parameters.dict()
#             save_args = orbax_utils.save_args_from_target(ckpt)
#             orbax_checkpointer.save(
#                 os.path.join(training_parameters.checkpoint_path, f"epoch-{epoch}.ckpt"),
#                 ckpt,
#                 save_args=save_args,
#                 force=True,
#             )
#         key, subkey = jax.random.split(key)
#         batch_generator = generate_batch(
#             key=subkey,
#             data=(data.x, data.y),
#             batch_size=training_parameters.batch_size,
#             shuffle=True,
#             drop_last=False,
#         )
#         data_batch = next(batch_generator, None)
#         while data_batch is not None:
#             x_batch, y_batch = data_batch
#             gradients = jax.grad(
#                 lambda gp_parameters_dict: nll_loss.calculate_empirical_risk(
#                     parameters=gp_parameters_dict,
#                     x=x_batch,
#                     y=y_batch,
#                 )
#             )(gp_parameters.dict())
#             updates, opt_state = training_parameters.optimiser_scheme.update(gradients, opt_state)
#             gp_parameters = gp.generate_parameters(
#                 optax.apply_updates(gp_parameters.dict(), updates)
#             )
#             data_batch = next(batch_generator, None)
#         losses.append(
#             nll_loss.calculate_empirical_risk(
#                 parameters=gp_parameters,
#                 x=data.x,
#                 y=data.y,
#             )
#         )
#         if losses[-1] < nll_break_condition:
#             break
#     return gp_parameters, losses
#
#
# def train_tempered_nll(
#     gp: GPBase,
#     gp_parameters: GPBaseParameters,
#     base_gp_parameters: GPBaseParameters,
#     data: Data,
#     training_parameters: TrainingParameters,
# ) -> Tuple[GPBaseParameters, List[float]]:
#     optimizer = optax.adabelief(training_parameters.learning_rate)
#     losses = []
#     opt_state = optimizer.init(gp_parameters.kernel.dict())
#     nll_loss = NegativeLogLikelihood(gp=gp)
#     key = training_parameters.key
#     for epoch in tqdm(range(training_parameters.number_of_epochs)):
#         if epoch % training_parameters.save_checkpoint_frequency == 0:
#             ckpt = gp_parameters.dict()
#             save_args = orbax_utils.save_args_from_target(ckpt)
#             orbax_checkpointer.save(
#                 os.path.join(training_parameters.checkpoint_path, f"epoch-{epoch}.ckpt"),
#                 ckpt,
#                 save_args=save_args,
#                 force=True,
#             )
#         key, subkey = jax.random.split(key)
#         batch_generator = generate_batch(
#             key=subkey,
#             data=(data.x, data.y),
#             batch_size=training_parameters.batch_size,
#             shuffle=True,
#             drop_last=False,
#         )
#         data_batch = next(batch_generator, None)
#         while data_batch is not None:
#             x_batch, y_batch = data_batch
#             gradients = jax.grad(
#                 lambda tempered_gp_parameters_kernel_dict: nll_loss.calculate_empirical_risk(
#                     parameters=gp.Parameters(
#                         log_observation_noise=base_gp_parameters.log_observation_noise,
#                         mean=base_gp_parameters.mean,
#                         kernel=TemperedKernelParameters(
#                             **tempered_gp_parameters_kernel_dict
#                         ),
#                     ).dict(),
#                     x=x_batch,
#                     y=y_batch,
#                 )
#             )(gp_parameters.kernel.dict())
#             updates, opt_state = optimizer.update(gradients, opt_state)
#             tempered_gp_kernel_parameters = gp.kernel.generate_parameters(
#                 optax.apply_updates(gp_parameters.kernel.dict(), updates)
#             )
#             gp_parameters = gp.Parameters(
#                 log_observation_noise=base_gp_parameters.log_observation_noise,
#                 mean=base_gp_parameters.mean,
#                 kernel=tempered_gp_kernel_parameters,
#             )
#             data_batch = next(batch_generator, None)
#         losses.append(
#             nll_loss.calculate_empirical_risk(
#                 parameters=gp_parameters,
#                 x=data.x,
#                 y=data.y,
#             )
#         )
#     return gp_parameters, losses
#
#
# def train_gvi(
#     gp_parameters: GPBaseParameters,
#     gvi: GeneralisedVariationalInference,
#     data: Data,
#     training_parameters: TrainingParameters,
# ) -> Tuple[GPBaseParameters, List[float], List[float], List[float]]:
#     optimizer = optax.adabelief(training_parameters.learning_rate)
#     gvi_losses = []
#     emp_risk_losses = []
#     reg_losses = []
#     opt_state = optimizer.init(gp_parameters.dict())
#     key = training_parameters.key
#     for epoch in tqdm(range(training_parameters.number_of_epochs)):
#         if epoch % training_parameters.save_checkpoint_frequency == 0:
#             ckpt = gp_parameters.dict()
#             save_args = orbax_utils.save_args_from_target(ckpt)
#             orbax_checkpointer.save(
#                 os.path.join(training_parameters.checkpoint_path, f"epoch-{epoch}.ckpt"),
#                 ckpt,
#                 save_args=save_args,
#                 force=True,
#             )
#         key, subkey = jax.random.split(key)
#         batch_generator = generate_batch(
#             key=subkey,
#             data=(data.x, data.y),
#             batch_size=training_parameters.batch_size,
#             shuffle=True,
#             drop_last=False,
#         )
#         data_batch = next(batch_generator, None)
#         while data_batch is not None:
#             x_batch, y_batch = data_batch
#             gradients = jax.grad(
#                 lambda gp_parameters_dict: gvi.calculate_loss(
#                     parameters=gp_parameters_dict,
#                     x=x_batch,
#                     y=y_batch,
#                 )
#             )(gp_parameters.dict())
#             updates, opt_state = optimizer.update(gradients, opt_state)
#             gp_parameters = gvi.regularisation.gp.generate_parameters(
#                 optax.apply_updates(gp_parameters.dict(), updates)
#             )
#             data_batch = next(batch_generator, None)
#         gvi_losses.append(
#             gvi.calculate_loss(
#                 parameters=gp_parameters,
#                 x=data.x,
#                 y=data.y,
#             )
#         )
#         emp_risk_losses.append(
#             gvi.empirical_risk.calculate_empirical_risk(
#                 parameters=gp_parameters,
#                 x=data.x,
#                 y=data.y,
#             )
#         )
#         reg_losses.append(
#             gvi.regularisation.calculate_regularisation(
#                 parameters=gp_parameters,
#                 x=data.x,
#             )
#         )
#     return gp_parameters, gvi_losses, emp_risk_losses, reg_losses
