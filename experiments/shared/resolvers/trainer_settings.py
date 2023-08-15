from typing import Dict, Union

from flax.core.frozen_dict import FrozenDict

from experiments.shared.trainer import TrainerSettings


def trainer_settings_resolver(
    trainer_settings_config: Union[FrozenDict, Dict]
) -> TrainerSettings:
    assert (
        "seed" in trainer_settings_config
    ), "Seed must be specified for trainer settings"
    assert (
        "optimiser_scheme" in trainer_settings_config
    ), "Optimiser scheme must be specified."
    assert (
        "learning_rate" in trainer_settings_config
    ), "Learning rate must be specified."
    assert (
        "number_of_epochs" in trainer_settings_config
    ), "Number of epochs must be specified."
    assert "batch_size" in trainer_settings_config, "Batch size must be specified."
    assert (
        "batch_shuffle" in trainer_settings_config
    ), "Batch shuffle must be specified."
    assert (
        "batch_drop_last" in trainer_settings_config
    ), "Batch drop last must be specified."
    return TrainerSettings(
        seed=trainer_settings_config["seed"],
        optimiser_scheme=trainer_settings_config["optimiser_scheme"],
        learning_rate=trainer_settings_config["learning_rate"],
        number_of_epochs=trainer_settings_config["number_of_epochs"],
        batch_size=trainer_settings_config["batch_size"],
        batch_shuffle=trainer_settings_config["batch_shuffle"],
        batch_drop_last=trainer_settings_config["batch_drop_last"],
    )
