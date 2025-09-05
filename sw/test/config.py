from dataclasses import dataclass


@dataclass
class DataConfig:
    file_path: str = "/home/mehrab/Projects/Cogni-E-Spin-Inverse-Design/dataset/PVA-Ziabari-2009.csv"
    file_type: str = "csv"
    target_column: str = "diameter"


@dataclass
class ModelConfig:
    n_estimators: int = 50
    colsample_bytree: float = 1.0
    learning_rate: float = 0.2
    max_depth: int = 3
    subsample: float = 0.6
    random_state: int = 42


@dataclass
class OptimizerConfig:
    n_iter: int = 100
    n_particles: int = 30
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    max_velocity: float = 0.2
    early_stop_patience: int = 10


@dataclass
class EvalConfig:
    method_name: str = "pso"
    noise_var: float = 0.01
    n_perturb: int = 10
    verbose: bool = True
