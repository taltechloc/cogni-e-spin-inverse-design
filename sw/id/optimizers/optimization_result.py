# optimizers/result.py
from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass
class OptimizationResult:
    """
    Generic container for storing optimizer results.
    """
    best_candidates: np.ndarray       # Best solution found
    best_prediction: Any          # Predicted value or objective function value
    cost_history: List[float]     # History of best costs over iterations
    n_iterations: int = 0         # Number of iterations completed
