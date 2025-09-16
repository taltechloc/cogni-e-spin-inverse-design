from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    # Define the hyperparameter grid for PSO
    param_grid = {
        "n_particles": [10],
        "w_max": [0.8],
        "w_min": [0.3, 0.4],
        "c1": [1.5, 2.0],
        "c2": [1.5, 2.0],
        "max_velocity": [0.2]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Particle Swarm Optimization",
        param_grid=param_grid
    )
