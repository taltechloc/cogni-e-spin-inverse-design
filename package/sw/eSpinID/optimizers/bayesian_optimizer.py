# optimizers/bayesian_optimizer.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from eSpinID.optimizers.base_optimizer import BaseOptimizer
from eSpinID.optimizers.optimization_result import OptimizationResult

BOConfig = {
    "n_init": None,
    "n_iter": None,
    "acquisition_function": None,
    "kappa": None,
    "xi": None,
    "early_stop_patience": None,
    "optimizer_type": "BO"
}


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimizer for inverse design: minimizes squared error between
    model prediction and a target value (i.e. find input x s.t. model(x) ≈ target).

    Notes:
    - We treat y = (pred - target)^2 as the objective to *minimize*.
    - Acquisition functions (UCB, EI, PI) are implemented in the convention that
      **larger acquisition values are better**; we therefore maximize acquisition
      and call a local optimizer on -acquisition.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.n_init = int(definition.get("n_init", 10))
        self.n_iter = int(definition.get("n_iter", 50))
        self.acquisition_function = definition.get("acquisition_function", "ucb")
        self.kappa = float(definition.get("kappa", 2.576))
        self.xi = float(definition.get("xi", 0.0))
        self.early_stop_patience = int(definition.get("early_stop_patience", 20))
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries], dtype=float)
        self.upper_bounds = np.array([b[1] for b in boundaries], dtype=float)

        # GP kernel (sane bounds but not extreme)
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e5)) \
                 + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))


    def _squared_error(self, x, target):
        """
        x: 1D array of shape (dim,) or 2D (1, dim)
        returns scalar squared error between model(x) and target.
        """
        x = np.array(x).reshape(1, -1)
        pred = np.asarray(self.objective.model.predict(x)).ravel()[0]
        return float((pred - target) ** 2)

    def optimize(self, target):
        """
        Run Bayesian optimization to find input(s) whose model prediction matches `target`.
        Returns an OptimizationResult-like object similar to your original.
        """

        # --- initial design (random uniform)
        X = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(self.n_init, self.dim))
        y = np.array([self._squared_error(x, target) for x in X], dtype=float)

        best_idx = int(np.argmin(y))
        best_solution = X[best_idx].copy()
        best_fitness = float(y[best_idx])  # minimum squared error

        cost_history = [best_fitness]
        top_candidates = [(best_fitness, best_solution.copy())]
        no_improvement_counter = 0

        bounds = list(zip(self.lower_bounds, self.upper_bounds))

        for it in range(self.n_iter):
            # Fit GP on the current (X,y)
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(X, y)

            # Acquisition function: return scalar (higher is better)
            def acquisition(x_raw):
                x = np.array(x_raw).reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                mu = float(np.atleast_1d(mu)[0])
                sigma = float(np.atleast_1d(sigma)[0])
                sigma = max(sigma, 1e-9)

                acq_type = self.acquisition_function.upper()
                if acq_type == "UCB":
                    # For minimization of objective: we want low mu but high uncertainty.
                    # Define acquisition so larger = better: -mu + kappa*sigma
                    return -mu + self.kappa * sigma

                elif acq_type == "EI":
                    # Expected Improvement for minimization: improvement = best - mu - xi
                    improvement = best_fitness - mu - self.xi
                    if sigma <= 0:
                        return max(0.0, improvement)
                    z = improvement / sigma
                    return improvement * norm.cdf(z) + sigma * norm.pdf(z)

                elif acq_type == "PI":
                    improvement = best_fitness - mu - self.xi
                    if sigma <= 0:
                        return 1.0 if improvement > 0 else 0.0
                    z = improvement / sigma
                    return float(norm.cdf(z))

                else:
                    raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")

            # We will minimize -acquisition (to maximize acquisition). Use several restarts.
            def neg_acquisition(x):
                # minimize negative (scalar)
                return -float(acquisition(x))

            n_restarts = 8
            best_local_val = np.inf
            best_point = None

            for _ in range(n_restarts):
                x0 = np.random.uniform(self.lower_bounds, self.upper_bounds)
                try:
                    res = minimize(
                        neg_acquisition,
                        x0=x0,
                        bounds=bounds,
                        method="L-BFGS-B",
                        options={"maxiter": 200}
                    )
                except Exception:
                    continue

                if not res.success and res.fun is None:
                    continue

                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_point = res.x

            # Fallback: if optimizer failed, sample random candidate and evaluate acquisition
            if best_point is None:
                random_pts = np.random.uniform(self.lower_bounds, self.upper_bounds, size=(10, self.dim))
                acq_vals = [acquisition(r) for r in random_pts]
                best_point = random_pts[int(np.argmax(acq_vals))]

            next_point = np.array(best_point).reshape(self.dim,)
            next_value = self._squared_error(next_point, target)

            # Append new data (ensure shapes)
            X = np.vstack((X, next_point.reshape(1, -1)))
            y = np.append(y, next_value)

            # Update best
            if next_value < best_fitness - 1e-12:
                best_fitness = float(next_value)
                best_solution = next_point.copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            cost_history.append(best_fitness)
            top_candidates.append((float(next_value), next_point.copy()))
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

            if no_improvement_counter >= self.early_stop_patience:
                break

        predicted = float(np.asarray(self.objective.model.predict(best_solution.reshape(1, -1))).ravel()[0])
        top_positions = [p for c, p in top_candidates]

        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Iteration",
                ylabel="Squared Error",
                title="Bayesian Optimization Convergence",
                label="Best Cost"
            )
        }

        # Return a single best candidate (as 1D array) and other metadata
        return OptimizationResult(
            best_candidates=best_solution,       # 1D ndarray of length dim
            best_prediction=predicted,          # model(best_solution)
            cost_history=cost_history,
            top_candidates=top_positions,       # list of 1D ndarrays
            n_iterations=len(cost_history),
            plots_data=plots_data
        )
