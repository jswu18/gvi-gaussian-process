from typing import Dict, Tuple, Union

from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers.nn_mean_function import nn_mean_function_resolver
from experiments.shared.schemes import MeanScheme
from src.means import ConstantMean, CustomMean
from src.means.base import MeanBase, MeanBaseParameters


def mean_resolver(
    mean_config: Union[FrozenDict, Dict]
) -> Tuple[MeanBase, MeanBaseParameters]:
    assert "mean_scheme" in mean_config, "Mean scheme must be specified."
    assert "mean_kwargs" in mean_config, "Mean kwargs must be specified."
    assert "mean_parameters" in mean_config, "Mean parameters must be specified."
    mean_scheme: MeanScheme = mean_config["mean_scheme"]
    mean_kwargs: Union[FrozenDict, Dict] = mean_config["mean_kwargs"]
    mean_parameters_config: Union[FrozenDict, Dict] = mean_config["mean_parameters"]

    if mean_scheme == MeanScheme.constant:
        assert (
            "number_output_dimensions" in mean_kwargs
        ), "Number of output dimensions must be specified."
        mean = ConstantMean(
            number_output_dimensions=mean_kwargs["number_output_dimensions"]
        )
        assert "constant" in mean_parameters_config, "Constant must be specified."
        mean_parameters = mean.generate_parameters(
            {"constant": mean_parameters_config["constant"]}
        )
        return mean, mean_parameters
    if mean_scheme == MeanScheme.custom:
        assert (
            "nn_mean_function_kwargs" in mean_kwargs
        ), "Custom mean function kwargs must be specified."
        mean_function, mean_function_parameters = nn_mean_function_resolver(
            nn_mean_function_kwargs=mean_kwargs["nn_mean_function_kwargs"],
        )
        assert (
            "input_shape" in mean_kwargs["nn_mean_function_kwargs"]
        ), "Input shape must be specified."
        assert (
            "number_output_dimensions" in mean_kwargs
        ), "Number of output dimensions must be specified."
        mean = CustomMean(
            mean_function=mean_function,
            number_output_dimensions=mean_kwargs["number_output_dimensions"],
            preprocess_function=lambda x: x.reshape(
                -1,
                *mean_kwargs["nn_mean_function_kwargs"]["input_shape"],
            ),
        )
        mean_parameters = mean.generate_parameters(
            {
                "custom": mean_function_parameters,
            }
        )
        return mean, mean_parameters
    else:
        raise ValueError(f"Unknown mean scheme: {mean_scheme}.")
