# pipeline_factory.py

from id.pipeline import Pipeline
from id.objective.objective_type import ObjectiveType
from id.optimizers.optimizer_type import OptimizerType


class PipelineFactory:
    @staticmethod
    def create_pipeline(definition, data_x, model):
        # 1. Create boundaries
        boundaries = PipelineFactory._create_boundaries(
            data_x, definition.get("boundaries", {})
        )

        # 2. Create objective
        objective_function = PipelineFactory._create_objective(definition.get("objective"), model)

        # 3. Create optimizer
        optimizer_definition = definition.get("optimizer", {})
        optimizer_type_str = optimizer_definition.get("optimizer_type")
        optimizer_type = OptimizerType[optimizer_type_str]

        optimizer = optimizer_type.create(
            definition=optimizer_definition,
            objective=objective_function,
            boundaries=boundaries
        )

        # 4. Return DI-based pipeline
        return Pipeline(
            optimizer=optimizer,
            objective_function=objective_function,
            boundaries=boundaries,
        )

    @staticmethod
    def _create_boundaries(data_x, boundary_definition):
        method = boundary_definition.get("method", "minmax")
        if method == "minmax":
            return [(data_x[col].min(), data_x[col].max()) for col in data_x.columns]
        elif method == "fixed":
            return boundary_definition["values"]  # expects list of (min, max)
        else:
            raise ValueError(f"Unknown boundary method: {method}")

    @staticmethod
    def _create_objective(objective_definition, model):
        objective_type_str = objective_definition.get("objective_type", "SURROGATE").upper()
        objective_type = ObjectiveType[objective_type_str]
        return objective_type.create(model, **objective_definition.get("params", {}))
