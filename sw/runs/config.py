from id.models.xgboost import XGBoostConfig
from id.optimizers.pso import PSOConfig


class Global:
    surrogate_model_type = "xgboost"
    optimizer_type = "PSO"


class DataConfig:
    file_path = "../../data/PVA-Ziabari-2009.csv"
    file_type = "csv"
    target_column = "diameter"


xgb_config = XGBoostConfig(
    n_estimators=100,
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    random_state=42
)

pso_config = PSOConfig(
    n_iter=100,             # number of iterations
    n_particles=30,         # number of particles
    w_max=0.9,              # max inertia weight
    w_min=0.4,              # min inertia weight
    c1=2.0,                 # cognitive coefficient
    c2=2.0,                 # social coefficient
    max_velocity=0.5,       # maximum velocity of particles
    early_stop_patience=10  # patience for early stopping
)