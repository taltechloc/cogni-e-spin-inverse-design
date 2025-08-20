from dataclasses import dataclass
from typing import Any, List

@dataclass
class InverseDesignResult:
    targets: List[Any]
    predictions: List[Any]
    candidates: List[Any]
    cost_histories: List[Any]
    model: Any

class InverseDesignPipeline:
    def __init__(self, config):
        self.method_name = config.method_name

    def run(self, method_func, model, X_train, y_train, X_val, y_val, boundaries):
        model.train(X_train, y_train)

        targets, preds, candidates, cost_histories = [], [], [], []

        for target in y_val.values:
            if self.method_name == "knn":
                best_input, pred, cost_hist = method_func(model, target, X_train.values, y_train.values)
            else:
                best_input, pred, cost_hist = method_func(model, target, boundaries)

            targets.append(target)
            preds.append(pred)
            candidates.append(best_input)
            cost_histories.append(cost_hist)

        return InverseDesignResult(
            targets=targets,
            predictions=preds,
            candidates=candidates,
            cost_histories=cost_histories,
            model=model
        )
