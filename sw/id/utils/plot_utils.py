import numpy as np
import matplotlib.pyplot as plt

def plot_target_vs_prediction_per_fold(results_dict, method_name, n_folds=5, save_path=None):
    predictions = np.array(results_dict["predictions"])
    targets = np.array(results_dict["targets"])
    fold_preds = np.array_split(predictions, n_folds)
    fold_targets = np.array_split(targets, n_folds)
    colors = plt.cm.tab10.colors

    plt.figure(figsize=(8, 6))
    for i, (tgt, pred) in enumerate(zip(fold_targets, fold_preds), 1):
        plt.scatter(tgt, pred, color=colors[i-1], alpha=0.7, label=f'Fold {i}')
        plt.plot([min(tgt), max(tgt)], [min(tgt), max(tgt)], 'k--', linewidth=1)
    plt.title(f'{method_name} Predictions vs Target Diameters per Fold')
    plt.xlabel('Target Diameter')
    plt.ylabel('Predicted Diameter')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Fold prediction plot saved to {save_path}")
    plt.close()


def plot_target_vs_prediction_overall(results_dict, method_name, save_path=None):
    all_targets = np.array(results_dict["targets"])
    all_preds = np.array(results_dict["predictions"])
    min_len = min(len(all_targets), len(all_preds))
    all_targets, all_preds = all_targets[:min_len], all_preds[:min_len]

    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_preds, alpha=0.7, edgecolors='k')
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.title(f"{method_name} Predictions vs Target")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_cost_trajectories(fold_cost_histories_all, method_name, save_path=None):
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 6))
    for fold_idx, avg_cost in enumerate(fold_cost_histories_all, 1):
        plt.plot(avg_cost, label=f'Fold {fold_idx}', color=colors[(fold_idx-1) % 10], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost (Squared Error)')
    plt.title(f'{method_name} Cost Function Trajectories (Fold-level)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
