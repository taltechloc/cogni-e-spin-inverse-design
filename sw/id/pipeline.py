import os

import numpy as np
import pandas as pd

from id.objective.surrogate_objective import SurrogateObjective
from id.optimizers.optimizer_type import OptimizerType


class Pipeline:
    def __init__(self, optimizer_config, data_x, model):
        self.boundaries = self._create_boundaries(data_x)
        self.objective_function = self._create_objective(model)
        self.optimizer = self._initialize_optimizer(optimizer_config, self.objective_function, self.boundaries)
        self.result = None

    @staticmethod
    def _initialize_optimizer(optimizer_definition, objective_function, boundaries):
        optimizer_type_str = getattr(optimizer_definition, "optimizer_type", None)
        if optimizer_type_str is None:
            raise ValueError("OptimizerConfig must have a 'optimizer_type' attribute.")

        try:
            optimizer_type = OptimizerType[optimizer_type_str]
        except KeyError:
            raise ValueError(f"Unknown optimizer type: {optimizer_type_str}")

        optimizer = optimizer_type.create(
            config=optimizer_definition,
            objective=objective_function,
            boundaries=boundaries
        )
        return optimizer

    @staticmethod
    def _create_objective(model):
        objective_function = SurrogateObjective(model)
        return objective_function

    @staticmethod
    def _create_boundaries(data_x):
        boundaries = [(data_x[col].min(), data_x[col].max()) for col in data_x.columns]
        return boundaries

    def run(self, target):
        self.result = self.optimizer.optimize(target)
        return self.result

    def __str__(self):
        if self.result is None:
            return "Pipeline result: Not yet run."
        return (
            f"Pipeline result:\n"
            f"  Best Candidates: {self.result.best_candidates}\n"
            f"  Best Prediction: {self.result.best_prediction}\n"
            f"  Iterations: {self.result.n_iterations}"
        )
