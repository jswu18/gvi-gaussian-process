from typing import Dict, Tuple, Union

from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers import neural_network_mean_function_resolver
from experiments.shared.schemes import MeanScheme
from src.means import ConstantMean, CustomMean
from src.means.base import MeanBase, MeanBaseParameters


def mean_resolver(
    mean_scheme: MeanScheme,
    mean_kwargs: Union[FrozenDict, Dict],
    mean_parameters: Union[FrozenDict, Dict],
) -> Tuple[MeanBase, MeanBaseParameters]:
    if mean_scheme == MeanScheme.constant:
        assert (
            "number_output_dimensions" in mean_kwargs
        ), "Number of output dimensions must be specified."
        mean = ConstantMean(
            number_output_dimensions=mean_kwargs["number_output_dimensions"]
        )
        assert "constant" in mean_parameters, "Constant must be specified."
        mean_parameters = mean.generate_parameters(
            {"constant": mean_parameters["constant"]}
        )
        return mean, mean_parameters
    if mean_scheme == MeanScheme.custom:
        assert (
            "neural_network_function_kwargs" in mean_kwargs["mean_function"]
        ), "Custom mean function kwargs must be specified."
        mean_function, mean_function_parameters = neural_network_mean_function_resolver(
            neural_network_function_kwargs=mean_kwargs["mean_function"][
                "neural_network_function_kwargs"
            ],
        )
        mean = CustomMean(
            mean_function=mean_function,
        )
        mean_parameters = mean.generate_parameters(
            {
                "custom": mean_function_parameters,
            }
        )
        return mean, mean_parameters
    else:
        raise ValueError(f"Unknown mean scheme: {mean_scheme}.")
