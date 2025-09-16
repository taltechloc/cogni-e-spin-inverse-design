import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

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

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.n_init = definition.get("n_init", 10)
        self.n_iter = definition.get("n_iter", 50)
        self.acquisition_function = definition.get("acquisition_function", "ucb")
        self.kappa = definition.get("kappa", 2.576)
        self.xi = definition.get("xi", 0.0)
        self.early_stop_patience = definition.get("early_stop_patience", 20)
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries])
        self.upper_bounds = np.array([b[1] for b in boundaries])

        # GP kernel
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    def optimize(self, target):
        def obj_fun(x):
            x = np.array(x).reshape(1, -1)
            pred = self.objective.model.predict(x)[0]
            return (pred - target) ** 2

        # Initial samples
        X = np.random.uniform(self.lower_bounds, self.upper_bounds,
                              size=(self.n_init, self.dim))
        y = np.array([obj_fun(x) for x in X])

        best_idx = np.argmin(y)
        best_solution = X[best_idx].copy()
        best_fitness = y[best_idx]

        cost_history = [best_fitness]
        top_candidates = [(best_fitness, best_solution.copy())]
        no_improvement_counter = 0

        for i in range(self.n_iter):
            # Fit GP
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
            gp.fit(X, y)

            # Acquisition function
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)

                if self.acquisition_function == "ucb":
                    return mu - self.kappa * sigma
                elif self.acquisition_function == "ei":
                    improvement = best_fitness - mu - self.xi
                    z = improvement / (sigma + 1e-9)
                    return improvement * norm.cdf(z) + sigma * norm.pdf(z)
                else:  # poi
                    improvement = best_fitness - mu - self.xi
                    z = improvement / (sigma + 1e-9)
                    return norm.cdf(z)

            # Optimize acquisition function
            from scipy.optimize import minimize

            def neg_acquisition(x):
                return -acquisition(x)

            result = minimize(neg_acquisition,
                              x0=np.random.uniform(self.lower_bounds, self.upper_bounds),
                              bounds=list(zip(self.lower_bounds, self.upper_bounds)),
                              method='L-BFGS-B')

            next_point = result.x
            next_value = obj_fun(next_point)

            # Update data
            X = np.vstack((X, next_point))
            y = np.append(y, next_value)

            # Update best
            if next_value < best_fitness:
                best_fitness = next_value
                best_solution = next_point.copy()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            cost_history.append(best_fitness)
            top_candidates.append((next_value, next_point.copy()))
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

            if no_improvement_counter >= self.early_stop_patience:
                break

        predicted = self.objective.model.predict(best_solution.reshape(1, -1))[0]
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
        return OptimizationResult(
            best_candidates=best_solution,
            best_prediction=predicted,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )
