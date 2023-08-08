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

from experiments import resolvers
from experiments.data import Data
from experiments.schemes import Optimiser
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

    def train(
        self,
        trainer_settings: TrainerSettings,
        parameters: ModuleParameters,
        data: Data,
        loss_function: Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], float],
        disable_tqdm: bool = False,
    ) -> Tuple[ModuleParameters, List[Dict[str, float]]]:
        post_epoch_history = []
        optimiser = resolvers.optimiser(
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
