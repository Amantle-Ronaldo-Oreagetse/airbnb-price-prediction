from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "city_datasets"
MODEL_DIR = PROJECT_DIR / "models" / "final_models"

OUTPUT_DIR = PROJECT_DIR / "reports" / "error_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ANALYSIS SETTINGS
# ============================================================

RANDOM_STATE = 42

MAX_IMPORTANCE_SAMPLES = 500
IMPORTANCE_REPEATS = 3

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "bedrooms_missing",
    "minimum_nights",
    "maximum_nights",
    "amenities_count",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "room_type",
    "neighbourhood",
    "instant_bookable",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 3. STORAGE FOR CONSOLIDATED RESULTS
# ============================================================

city_summaries = []
error_by_price_results = []
feature_importance_results = []
largest_errors_results = []


# ============================================================
# 4. PROCESS EACH CITY
# ============================================================

city_directories = sorted(
    [path for path in DATA_DIR.iterdir() if path.is_dir()]
)

print(f"Cities discovered: {len(city_directories)}")


for city_dir in city_directories:

    city = city_dir.name

    print("\n" + "=" * 70)
    print(f"FINAL ERROR ANALYSIS: {city.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data and saved final model
    # --------------------------------------------------------

    test_path = city_dir / "test.csv"

    model_path = MODEL_DIR / f"{city}_final_model.joblib"

    if not test_path.exists():
        print(f"Test dataset missing for {city}. Skipping.")
        continue

    if not model_path.exists():
        print(f"Final model missing for {city}. Skipping.")
        continue

    test_df = pd.read_csv(test_path)

    model_package = joblib.load(model_path)

    model = model_package["model"]

    model_features = model_package.get("features", FEATURES)

    X_test = test_df[model_features].copy()

    y_actual = test_df["price"].to_numpy()

    # --------------------------------------------------------
    # Reproduce predictions using the saved model
    # --------------------------------------------------------

    y_predicted = model.predict(X_test)

    y_predicted = np.maximum(y_predicted, 0)

    residuals = y_actual - y_predicted

    absolute_errors = np.abs(residuals)

    mae = mean_absolute_error(y_actual, y_predicted)

    rmse = np.sqrt(np.mean(residuals ** 2))

    r2 = 1 - (
        np.sum(residuals ** 2)
        /
        np.sum((y_actual - np.mean(y_actual)) ** 2)
    )

    # --------------------------------------------------------
    # Error summary
    # --------------------------------------------------------

    summary = {
        "city": city,
        "test_records": len(y_actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_error": np.mean(residuals),
        "median_absolute_error": np.median(absolute_errors),
        "90th_percentile_absolute_error": np.percentile(
            absolute_errors, 90
        ),
        "95th_percentile_absolute_error": np.percentile(
            absolute_errors, 95
        ),
        "maximum_absolute_error": np.max(absolute_errors),
    }

    city_summaries.append(summary)

    print(f"Test records: {len(y_actual)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Mean residual: {np.mean(residuals):.3f}")

    # --------------------------------------------------------
    # Create detailed prediction results
    # --------------------------------------------------------

    error_df = pd.DataFrame({
        "actual_price": y_actual,
        "predicted_price": y_predicted,
        "residual": residuals,
        "absolute_error": absolute_errors,
    })

    if "listing_id" in test_df.columns:
        error_df["listing_id"] = test_df["listing_id"].values

    if "host_id" in test_df.columns:
        error_df["host_id"] = test_df["host_id"].values

    error_df["city"] = city

    # --------------------------------------------------------
    # Save the largest prediction errors
    # --------------------------------------------------------

    largest_errors = error_df.nlargest(
        20, "absolute_error"
    ).copy()

    largest_errors_results.append(largest_errors)

    # --------------------------------------------------------
    # Analyse errors across actual price ranges
    # --------------------------------------------------------

    error_df["price_range"] = pd.qcut(
        error_df["actual_price"],
        q=4,
        duplicates="drop"
    )

    price_error_summary = (
        error_df.groupby("price_range", observed=True)
        .agg(
            listing_count=("actual_price", "count"),
            mean_actual_price=("actual_price", "mean"),
            mean_predicted_price=("predicted_price", "mean"),
            MAE=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
        )
        .reset_index()
    )

    price_error_summary["city"] = city

    error_by_price_results.append(price_error_summary)

    # --------------------------------------------------------
    # Save individual error analysis data
    # --------------------------------------------------------

    error_df.to_csv(
        OUTPUT_DIR / f"{city}_prediction_errors.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot 1: Actual vs predicted prices
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_actual,
        y_predicted,
        alpha=0.25
    )

    plot_min = min(y_actual.min(), y_predicted.min())
    plot_max = max(y_actual.max(), y_predicted.max())

    plt.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(f"{city}: Actual vs Predicted Prices")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{city}_actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Residual distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=50
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")

    plt.title(f"{city}: Residual Distribution")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{city}_residual_distribution.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Residuals against predicted prices
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_predicted,
        residuals,
        alpha=0.25
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")

    plt.title(f"{city}: Residuals vs Predicted Prices")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{city}_residuals_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 5. PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nCalculating permutation feature importance...")

    sample_size = min(
        MAX_IMPORTANCE_SAMPLES,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    sample_indices = X_sample.index

    y_sample = pd.Series(
        y_actual,
        index=test_df.index
    ).loc[sample_indices].to_numpy()

    importance = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": model_features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })

    importance_df["city"] = city

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    )

    feature_importance_results.append(importance_df)

    # --------------------------------------------------------
    # Plot 4: Feature importance
    # --------------------------------------------------------

    top_features = importance_df.head(10).sort_values(
        "importance_mean"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"]
    )

    plt.xlabel("Decrease in Model Performance (MAE)")
    plt.ylabel("Feature")

    plt.title(f"{city}: Top 10 Permutation Importances")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{city}_feature_importance.png",
        dpi=150
    )

    plt.close()

    print("\nTop 5 influential features:")

    print(
        importance_df[
            ["feature", "importance_mean"]
        ].head(5).to_string(index=False)
    )


# ============================================================
# 6. SAVE CONSOLIDATED REPORTS
# ============================================================

if city_summaries:

    summary_df = pd.DataFrame(city_summaries)

    summary_df.to_csv(
        OUTPUT_DIR / "error_analysis_summary.csv",
        index=False
    )

if error_by_price_results:

    price_df = pd.concat(
        error_by_price_results,
        ignore_index=True
    )

    price_df.to_csv(
        OUTPUT_DIR / "error_by_price_range.csv",
        index=False
    )

if feature_importance_results:

    importance_df = pd.concat(
        feature_importance_results,
        ignore_index=True
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance_by_city.csv",
        index=False
    )

if largest_errors_results:

    largest_errors_df = pd.concat(
        largest_errors_results,
        ignore_index=True
    )

    largest_errors_df.to_csv(
        OUTPUT_DIR / "largest_prediction_errors.csv",
        index=False
    )


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STAGE 4.4 COMPLETE")
print("=" * 75)

if city_summaries:

    print(
        pd.DataFrame(city_summaries)[
            [
                "city",
                "MAE",
                "RMSE",
                "R2",
                "mean_error",
                "95th_percentile_absolute_error",
            ]
        ].round(3).to_string(index=False)
    )

print("\nError analysis saved to:")
print(OUTPUT_DIR)

print("\nFigures saved to:")
print(FIGURE_DIR)