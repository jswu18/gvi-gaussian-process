from typing import Dict, Tuple, Union

from flax.core.frozen_dict import FrozenDict

from experiments.shared.resolvers.nn_function import nn_function_resolver
from experiments.shared.schemas import MeanSchema
from src.means import ConstantMean, CustomMean
from src.means.base import MeanBase, MeanBaseParameters


def mean_resolver(
    mean_config: Union[FrozenDict, Dict],
    data_dimension: int,
) -> Tuple[MeanBase, MeanBaseParameters]:
    assert "mean_schema" in mean_config, "Mean schema must be specified."
    assert "mean_kwargs" in mean_config, "Mean kwargs must be specified."
    assert "mean_parameters" in mean_config, "Mean parameters must be specified."
    mean_schema: MeanSchema = mean_config["mean_schema"]
    mean_kwargs: Union[FrozenDict, Dict] = mean_config["mean_kwargs"]
    mean_parameters_config: Union[FrozenDict, Dict] = mean_config["mean_parameters"]

    if mean_schema == MeanSchema.constant:
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
    if mean_schema == MeanSchema.custom:
        assert (
            "nn_function_kwargs" in mean_kwargs
        ), "Custom mean function kwargs must be specified."
        mean_function, mean_function_parameters = nn_function_resolver(
            nn_function_kwargs=mean_kwargs["nn_function_kwargs"],
            data_dimension=data_dimension,
        )
        assert (
            "number_output_dimensions" in mean_kwargs
        ), "Number of output dimensions must be specified."
        if "input_shape" in mean_kwargs["nn_function_kwargs"]:

            def preprocess_function(x):
                return x.reshape(
                    -1,
                    *mean_kwargs["nn_function_kwargs"]["input_shape"],
                )

        else:

            def preprocess_function(x):
                return x.reshape(
                    -1,
                    data_dimension,
                )

        mean = CustomMean(
            mean_function=mean_function,
            number_output_dimensions=mean_kwargs["number_output_dimensions"],
            preprocess_function=preprocess_function,
        )
        mean_parameters = mean.generate_parameters(
            {
                "custom": mean_function_parameters,
            }
        )
        return mean, mean_parameters
    raise ValueError(f"Unknown mean schema: {mean_schema}.")
