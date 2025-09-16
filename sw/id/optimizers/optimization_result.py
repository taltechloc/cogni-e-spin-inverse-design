# optimizers/optimization_result.py
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class OptimizationResult:
    """
    Generic container for storing optimizer results.
    """
    best_candidates: np.ndarray
    best_prediction: Any
    cost_history: List[float]
    top_candidates: Optional[List[np.ndarray]]
    n_iterations: int = 0
    plots_data: Optional[dict[str, Any]] = None
