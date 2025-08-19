import numpy as np
from id.data.data_loader import DataLoader
from InverseDesign.model_trainer import ModelTrainer
from InverseDesign.optimizers.pso import PSOOptimizer


class InverseDesignPipeline:
    def __init__(self, data_path, target_col, optimizer_class=PSOOptimizer, model_params=None, optimizer_params=None):
        self.data_loader = DataLoader(data_path)
        self.target_col = target_col
        self.model_params = model_params or {}
        self.optimizer_params = optimizer_params or {}
        self.optimizer_class = optimizer_class
        self.model = None
        self.X = None
        self.y = None

    def load_data(self):
        df = self.data_loader.load()
        self.X = df.drop(columns=[self.target_col])
        self.y = df[self.target_col]

    def train_model(self, X_train, y_train):
        trainer = ModelTrainer(**self.model_params)
        self.model = trainer.train(X_train, y_train)
        return self.model

    def run_optimizer(self, target, boundaries):
        if self.model is None:
            raise ValueError("Model must be trained before running optimizer.")

        optimizer = self.optimizer_class(self.model, **self.optimizer_params)
        best_input, pred, cost_history = optimizer.run(target, boundaries)
        return best_input, pred, cost_history

    def get_features_targets(self):
        return self.X, self.y
