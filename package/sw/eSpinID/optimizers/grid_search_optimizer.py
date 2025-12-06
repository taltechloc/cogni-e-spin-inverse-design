# optimizers/grid_search_optimizer.py
import numpy as np
import warnings
from itertools import product
from math import prod

from eSpinID.optimizers.base_optimizer import BaseOptimizer
from eSpinID.optimizers.optimization_result import OptimizationResult

GridSearchConfig = {
    "optimizer_type": "GridSearch",
    "param_levels": [
        [8, 9, 10, 12],            # Solution Concentration
        [10, 12.5, 15, 20],        # Tip–Collector Distance
        [15, 22.5, 20, 25],        # Applied Voltage
        [0.2, 0.25, 0.3, 0.4]      # Feed Rate
    ],
    "replicas": 3,
    "jitter": 0.05,
    "randomize_order": False,
    "seed": None,
    "replica_strategy": "mean",
    "steps_per_param": 5,
    "param_types": None,   # optional: list like ["continuous", "integer", ...]
    "round_tol": 12,       # decimals used for uniqueness checks
    "materialize_threshold": 1_000_000  # max grid size to materialize for shuffle
}


class GridSearchOptimizer(BaseOptimizer):
    """
    Improved grid search optimizer with:
      - validation and clipping of provided param_levels
      - optional param_types (continuous/integer)
      - batch predictions for speed
      - constant-parameter handling (no jitter for constants)
      - deterministic RNG via seed
      - configurable uniqueness rounding tolerance
      - memory-safe handling of very large grids (avoid materializing huge lists)
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)

        # basic config + sensible defaults
        self.param_levels = definition.get("param_levels", None)
        self.dim = len(boundaries)
        if self.param_levels is not None:
            if len(self.param_levels) != self.dim:
                raise AssertionError("Levels must match number of parameters")

        # config
        self.replicas = max(1, int(definition.get("replicas", GridSearchConfig["replicas"])))
        self.jitter = float(definition.get("jitter", GridSearchConfig["jitter"]))
        self.randomize_order = bool(definition.get("randomize_order", GridSearchConfig["randomize_order"]))
        self.seed = definition.get("seed", GridSearchConfig["seed"])
        self.replica_strategy = definition.get("replica_strategy", GridSearchConfig["replica_strategy"])
        self.steps_per_param = int(definition.get("steps_per_param", GridSearchConfig["steps_per_param"]))
        # param types: "continuous" (default) or "integer"
        self.param_types = definition.get("param_types", GridSearchConfig["param_types"])
        if self.param_types is None:
            self.param_types = ["continuous"] * self.dim
        if len(self.param_types) != self.dim:
            raise ValueError("param_types length must match number of parameters")
        # uniqueness rounding tolerance
        self.round_tol = int(definition.get("round_tol", GridSearchConfig["round_tol"]))
        # materialization guard for randomization
        self.materialize_threshold = int(definition.get("materialize_threshold", GridSearchConfig["materialize_threshold"]))

        # validate replica_strategy
        if self.replica_strategy not in ("min", "mean"):
            raise ValueError(f"Unknown replica_strategy: {self.replica_strategy!r}. Use 'min' or 'mean'.")

        # bounds and derived ranges
        self.lower_bounds = np.array([b[0] for b in boundaries], dtype=float)
        self.upper_bounds = np.array([b[1] for b in boundaries], dtype=float)
        self.ranges = self.upper_bounds - self.lower_bounds

        # constant parameters (no jitter possible); store mask
        self.constant_mask = (np.isclose(self.ranges, 0.0))
        # avoid zero ranges breaking noise scale by leaving ranges unchanged for other computations,
        # but we'll explicitly avoid jittering constant dimensions.
        # For safety in scale computation use a surrogate scale when needed (not used for constants).
        self._noise_scale = self.ranges.copy()
        zero_mask = self._noise_scale == 0.0
        if np.any(zero_mask):
            # set a dummy scale of 1.0 for those dims to keep noise math stable,
            # but we will not use noise for those dimensions because of constant_mask.
            self._noise_scale[zero_mask] = 1.0

    def _validate_and_clip_levels(self, levels):
        """Clip provided param_levels to the boundaries and convert to numpy arrays."""
        grid_axes = []
        for i, lvl in enumerate(levels):
            arr = np.asarray(lvl, dtype=float)
            if np.any(arr < self.lower_bounds[i]) or np.any(arr > self.upper_bounds[i]):
                warnings.warn(
                    f"param_levels for parameter {i} contained values outside bounds; clipping to [{self.lower_bounds[i]}, {self.upper_bounds[i]}].",
                    UserWarning
                )
                arr = np.clip(arr, self.lower_bounds[i], self.upper_bounds[i])
            # If integer param, round and unique
            if self.param_types[i] == "integer":
                arr = np.unique(np.round(arr)).astype(float)
            else:
                # deduplicate approximate duplicates
                arr = np.unique(arr)
            grid_axes.append(arr.tolist())
        return grid_axes

    def _auto_grid_axes(self):
        """Generate grid axes using linspace; respect integer params."""
        axes = []
        for i in range(self.dim):
            if self.param_types[i] == "integer":
                # generate integer steps, ensure unique
                raw = np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.steps_per_param)
                ints = np.unique(np.round(raw)).astype(float)
                axes.append(ints.tolist())
            else:
                axes.append(np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.steps_per_param).tolist())
        return axes

    def _replicas_for_point(self, base_point, rng):
        """
        Yield unique replicas for base_point.
        - first yields the base_point itself (with integer rounding if needed)
        - then yields up to (replicas-1) uniformly jittered points clipped to bounds
        - uniqueness enforced by rounding to `self.round_tol` decimals
        - constant parameters are not jittered
        """
        base = np.asarray(base_point, dtype=float).copy()

        # enforce integer params on base
        for i, t in enumerate(self.param_types):
            if t == "integer":
                base[i] = np.round(base[i])

        seen = set()

        def add_and_yield(pt):
            # round to tolerance for uniqueness key
            key = tuple(np.round(pt, self.round_tol))
            if key in seen:
                return False
            seen.add(key)
            return True

        if add_and_yield(base):
            yield base.copy()

        # if no replicas requested or no jitter, return
        if self.replicas <= 1 or self.jitter <= 0.0:
            return

        attempts = 0
        max_attempts = max(50, self.replicas * 10)  # avoid infinite loops if uniqueness impossible

        while len(seen) < self.replicas and attempts < max_attempts:
            # uniform noise in [-1, +1] scaled by jitter and per-dimension noise scale
            u = rng.rand(self.dim) * 2.0 - 1.0
            noise = u * (self.jitter * self._noise_scale)
            # zero-out noise for constant dims
            noise[self.constant_mask] = 0.0
            cand = np.clip(base + noise, self.lower_bounds, self.upper_bounds)

            # enforce integer params on candidate
            for i, t in enumerate(self.param_types):
                if t == "integer":
                    cand[i] = np.round(cand[i])

            if add_and_yield(cand):
                yield cand
            attempts += 1

        if len(seen) < self.replicas:
            warnings.warn(
                f"Requested {self.replicas} unique replicas for point {base_point}, "
                f"but only generated {len(seen)} after {attempts} attempts.",
                UserWarning
            )

    def optimize(self, target):
        rng = np.random.RandomState(self.seed)

        # build grid axes
        if self.param_levels is not None:
            grid_axes = self._validate_and_clip_levels(self.param_levels)
        else:
            grid_axes = self._auto_grid_axes()

        # compute total grid size
        sizes = [len(ax) for ax in grid_axes]
        total_size = int(prod(sizes)) if sizes else 0

        # decide whether to materialize grid points for randomization
        if self.randomize_order:
            if total_size == 0:
                grid_points = []
            elif total_size <= self.materialize_threshold:
                # safe to materialize and shuffle
                grid_points = list(product(*grid_axes))
                rng.shuffle(grid_points)
            else:
                warnings.warn(
                    "Grid is very large (size={}); randomize_order=True requested but grid will not be shuffled to avoid excessive memory usage. Proceeding in deterministic (lexicographic) order.".format(total_size),
                    UserWarning
                )
                grid_points = product(*grid_axes)
        else:
            # avoid materializing the entire grid unless small and user wants it
            if total_size <= self.materialize_threshold:
                grid_points = list(product(*grid_axes))
            else:
                grid_points = product(*grid_axes)  # iterator

        cost_history = []
        best_cost = float('inf')
        best_input = None
        best_pred = None
        top_candidates = []

        # iterate grid (grid_points may be iterator or list)
        for point in grid_points:
            # create replicas first (list) so we can batch-predict
            replica_pts = list(self._replicas_for_point(point, rng))
            if len(replica_pts) == 0:
                continue

            X = np.vstack(replica_pts)  # (m, dim)

            # batch predict if possible (most models accept a 2D array)
            try:
                preds = self.objective.model.predict(X)
                preds = np.asarray(preds).reshape(-1)
                if preds.shape[0] != X.shape[0]:
                    # fallback: model returned a scalar or unexpected shape; try per-row predict
                    raise ValueError("Model returned wrong shape for batched predict")
            except Exception:
                # fallback to per-row predictions
                preds = []
                for row in X:
                    try:
                        p = self.objective.model.predict(row.reshape(1, -1))
                        # try to extract scalar
                        if np.ndim(p) == 0:
                            preds.append(float(p))
                        else:
                            preds.append(float(np.asarray(p).reshape(-1)[0]))
                    except Exception as e:
                        # re-raise with context
                        raise RuntimeError(f"Model predict failed on row {row}: {e}") from e
                preds = np.asarray(preds)

            costs = (preds - target) ** 2

            if self.replica_strategy == "min":
                idx_best = int(np.argmin(costs))
                cost_best = float(costs[idx_best])
                pred_best = float(preds[idx_best])
                point_best = replica_pts[idx_best]
            else:  # mean
                cost_best = float(np.mean(costs))
                pred_best = float(np.mean(preds))
                # choose replica closest to mean prediction for reporting
                idx_best = int(np.argmin(np.abs(preds - pred_best)))
                point_best = replica_pts[idx_best]

            cost_history.append(cost_best)

            if cost_best < best_cost:
                best_cost = cost_best
                best_input = np.array(point_best)
                best_pred = pred_best

            top_candidates.append((cost_best, np.array(point_best)))
            # keep only top 5 by cost
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

        # unique top positions (deduplicate near-duplicates)
        top_positions = []
        seen = set()
        for c, p in top_candidates:
            key = tuple(np.round(p, self.round_tol))
            if key not in seen:
                seen.add(key)
                top_positions.append(p)

        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Evaluation Index",
                ylabel="Squared Error",
                title="Grid Search Convergence",
                label="Cost per Evaluation"
            )
        }

        # wrap result: best_candidates (single best as array), best_prediction, etc.
        return OptimizationResult(
            best_candidates=best_input,
            best_prediction=best_pred,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )
