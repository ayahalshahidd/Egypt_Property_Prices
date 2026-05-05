from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


PALETTE = ["#3498db", "#e74c3c", "#16a085", "#94A3B8", "#2563EB", "#16A34A"]


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


def plot_market_overview(df_raw: DataFrame, output_path: str | Path, top_n: int = 8) -> None:
    category_counts = df_raw.groupBy("category").count().toPandas()
    top_types = (
        df_raw.groupBy("property_type")
        .agg(F.count("*").alias("count"))
        .orderBy(F.desc("count"))
        .limit(top_n)
        .toPandas()
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].pie(
        category_counts["count"],
        labels=[str(value).title() for value in category_counts["category"]],
        autopct="%1.1f%%",
        colors=PALETTE[: len(category_counts)],
        startangle=90,
    )
    axes[0].set_title("Buy vs Rent Split", fontsize=14, fontweight="bold")

    axes[1].barh(top_types["property_type"][::-1], top_types["count"][::-1], color=PALETTE[0])
    axes[1].set_title("Property Type Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_price_distributions(clean_df: DataFrame, output_path: str | Path) -> None:
    price_df = clean_df.select("category", _try_cast_double("price_egp").alias("price_egp")).dropna()
    buy_prices = (
        price_df.filter((F.col("category") == "buy") & (F.col("price_egp") < 50_000_000))
        .select((F.col("price_egp") / F.lit(1_000_000)).alias("price_m"))
        .toPandas()["price_m"]
    )
    rent_prices = (
        price_df.filter((F.col("category") == "rent") & (F.col("price_egp") < 500_000))
        .select((F.col("price_egp") / F.lit(1_000)).alias("price_k"))
        .toPandas()["price_k"]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(buy_prices, bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].set_title("Buy Price Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Price (Million EGP)")
    axes[1].hist(rent_prices, bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[1].set_title("Rent Price Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Price (Thousand EGP)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_property_map_sample(df_raw: DataFrame, output_path: str | Path, sample_size: int = 5_000) -> None:
    map_df = df_raw.select(
        "category",
        _try_cast_double("lat").alias("lat"),
        _try_cast_double("lon").alias("lon"),
    ).dropna(subset=["lat", "lon"])
    total_available = map_df.count()
    fraction = min(sample_size, total_available) / total_available if total_available else 0.0
    sample = map_df.sample(False, fraction, seed=42).toPandas()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for category, color in [("buy", PALETTE[4]), ("rent", PALETTE[5])]:
        category_points = sample[sample["category"] == category]
        ax.scatter(
            category_points["lon"],
            category_points["lat"],
            s=8,
            alpha=0.45,
            color=color,
            label=category.title(),
        )
    ax.set_title("Property Listings Map Sample (Egypt)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_price_area_relationships(clean_df: DataFrame, output_path: str | Path) -> None:
    base_df = clean_df.select(
        "category",
        _try_cast_double("area_value").alias("area_value"),
        _try_cast_double("price_egp").alias("price_egp"),
    ).dropna()
    buy = (
        base_df.filter(
            (F.col("category") == "buy")
            & (F.col("price_egp") < 50_000_000)
            & (F.col("area_value") < 1_000)
        )
        .sample(False, 0.2, seed=42)
        .toPandas()
    )
    rent = (
        base_df.filter(
            (F.col("category") == "rent")
            & (F.col("price_egp") < 100_000)
            & (F.col("area_value") < 500)
        )
        .sample(False, 0.5, seed=42)
        .toPandas()
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _scatter_with_trend(axes[0], buy, "area_value", "price_egp", "Buy Market: Price vs Area", PALETTE[4])
    _scatter_with_trend(axes[1], rent, "area_value", "price_egp", "Rent Market: Price vs Area", PALETTE[5])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_numeric_correlation_heatmap(clean_df: DataFrame, output_path: str | Path) -> None:
    numeric_cols = [
        "price_egp", "area_value", "bedrooms", "bathrooms",
        "lat", "lon", "price_per_sqm",
    ]
    corr_cols = [column_name for column_name in numeric_cols if column_name in clean_df.columns]
    corr_df = clean_df.select(
        *[_try_cast_double(column_name).alias(column_name) for column_name in corr_cols]
    ).dropna().toPandas()
    corr_matrix = corr_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    masked_corr = np.ma.array(corr_matrix.to_numpy(), mask=mask)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(masked_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8)
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


def _scatter_with_trend(ax, data: pd.DataFrame, x_col: str, y_col: str, title: str, color: str) -> None:
    ax.scatter(data[x_col], data[y_col], s=10, alpha=0.35, color=color)
    if len(data) > 1:
        slope, intercept = np.polyfit(data[x_col], data[y_col], 1)
        xs = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#111827", linewidth=1.2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Area (sqm)")
    ax.set_ylabel("Price (EGP)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))


def _try_cast_double(column_name: str) -> F.Column:
    return F.expr(f"try_cast(`{column_name}` as double)")


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
