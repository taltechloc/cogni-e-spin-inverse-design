import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MultipleLocator

def plot_target_vs_prediction(results_dict, method_name, r2, save_path=None, dpi=300):
    predictions = np.array(results_dict["predictions"])      # <- this is already the replica mean
    targets = np.array(results_dict["targets"])
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=100)

    # ----------------------------------------------------
    # SCATTER: Mean replica prediction (single layer)
    # ----------------------------------------------------
    ax.scatter(
        targets, predictions,
        color=colors[0],
        alpha=0.7,
        label='Mean prediction (replica mean)',   # <- UPDATED
        s=10,
        edgecolors='white',
        linewidth=0.1
    )

    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    margin = 0.05 * (max_val - min_val)

    # Ideal diagonal line
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        'k--',
        linewidth=0.8,
        alpha=0.8,
        label='Ideal'
    )

    # Trend line
    slope, intercept, _, _, _ = stats.linregress(targets, predictions)
    x_trend = np.array([min_val, max_val])
    y_trend = slope * x_trend + intercept
    ax.plot(
        x_trend, y_trend,
        'r-',
        linewidth=1.0,
        alpha=0.9,
        label=f'Fit (R²={r2:.3f})'
    )

    ax.set_aspect('equal')
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.4)

    ax.set_xlabel('Target Nanofiber Diameter (nm)', fontsize=8)
    ax.set_ylabel('Predicted Nanofiber  Diameter (nm)', fontsize=8)
    ax.set_title(f'{method_name}', fontsize=9, pad=6)

    # Keep the legend styling the SAME
    ax.legend(loc='upper left', frameon=True, fancybox=True,
              shadow=False, fontsize=6, framealpha=0.9)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=5)

    data_range = max_val - min_val
    if data_range <= 100:
        tick_spacing = 20
    elif data_range <= 200:
        tick_spacing = 40
    else:
        tick_spacing = 50

    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        print(f"Plot saved to {save_path}")

    plt.close()


# def plot_target_vs_prediction_per_fold(results_dict, method_name, n_folds=5, save_path=None):
#     predictions = np.array(results_dict["predictions"])
#     targets = np.array(results_dict["targets"])
#     fold_preds = np.array_split(predictions, n_folds)
#     fold_targets = np.array_split(targets, n_folds)
#     colors = plt.cm.tab10.colors
#
#     plt.figure(figsize=(8, 6))
#     for i, (tgt, pred) in enumerate(zip(fold_targets, fold_preds), 1):
#         plt.scatter(tgt, pred, color=colors[i-1], alpha=0.7, label=f'Fold {i}')
#         plt.plot([min(tgt), max(tgt)], [min(tgt), max(tgt)], 'k--', linewidth=1)
#     plt.title(f'{method_name} Predictions vs Target Diameters per Fold')
#     plt.xlabel('Target Diameter [nm]')
#     plt.ylabel('Predicted Diameter [nm]')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Fold prediction plot saved to {save_path}")
#     plt.close()


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
