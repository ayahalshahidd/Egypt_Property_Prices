from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PALETTE = ["#3498db", "#e74c3c", "#16a085", "#94A3B8"]


def plot_target_distribution(values: pd.Series, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_title("Price per sqm Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Price per sqm")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = [PALETTE[0] if i == 0 else PALETTE[3] for i in range(len(results_df))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(results_df["Model"][::-1], results_df["R2"][::-1], color=colors[::-1])
    axes[0].set_title("R2 Score (higher = better)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("R2")
    axes[1].barh(results_df["Model"][::-1], results_df["RMSE_EGP"][::-1], color=colors[::-1])
    axes[1].set_title("RMSE in EGP/sqm (lower = better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("RMSE (EGP/sqm)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true_log: pd.Series, y_pred_log: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, s=10, alpha=0.35, color=PALETTE[0])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Residuals vs Predicted Price per sqm", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted price per sqm")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_spark_feature_importance(model, features: list[str], output_path: str | Path, top_n: int = 20) -> None:
    estimator = model.stages[-1] if hasattr(model, "stages") else model
    if hasattr(estimator, "featureImportances"):
        scores = np.array(estimator.featureImportances.toArray())
        score_name = "importance"
        title = "Spark Feature Importance"
    elif hasattr(estimator, "coefficients"):
        scores = np.abs(np.array(estimator.coefficients.toArray()))
        score_name = "absolute_coefficient"
        title = "Spark Feature Coefficients"
    else:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame({"feature": features, score_name: scores})
    importance = importance.sort_values(score_name, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance["feature"][::-1], importance[score_name][::-1], color=PALETTE[2])
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
