from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "src"))

from importlib import import_module

inference = import_module("10_predict_listing_price")

load_city_model = inference.load_city_model
MODEL_FEATURES = inference.MODEL_FEATURES
CITY_CURRENCIES = inference.CITY_CURRENCIES


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "processed" / "city_datasets"

PREDICTION_DIR = PROJECT_DIR / "reports" / "final_test"

OUTPUT_DIR = PROJECT_DIR / "reports" / "inference_examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================

results = []

for city in CITY_CURRENCIES:

    print(f"\nChecking: {city}")

    city_slug = city.lower().replace(" ", "_")

    test_path = DATA_DIR / city_slug / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    test_data = pd.read_csv(test_path)

    # Find the original prediction file.
    prediction_files = list(
        PREDICTION_DIR.glob(f"{city_slug}_test_predictions.csv")
    )

    if len(prediction_files) != 1:

        raise FileNotFoundError(
            f"Expected one prediction file for {city}, "
            f"but found: {prediction_files}"
        )

    original_predictions = pd.read_csv(prediction_files[0])

    required_columns = [
        "listing_id",
        "predicted_price"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in original_predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {prediction_files[0]}: "
            f"{missing_columns}"
        )

    # Load the saved model.
    model = load_city_model(city)

    # Generate predictions again.
    regenerated = model.predict(
        test_data[MODEL_FEATURES]
    )

    regenerated = np.asarray(regenerated).reshape(-1)

    regenerated_data = pd.DataFrame({
        "listing_id": test_data["listing_id"],
        "regenerated_prediction": regenerated
    })

    # Match predictions using listing IDs.
    comparison = original_predictions[
        ["listing_id", "predicted_price"]
    ].merge(
        regenerated_data,
        on="listing_id",
        how="inner",
        validate="one_to_one"
    )

    if len(comparison) != len(test_data):

        raise ValueError(
            f"Prediction record mismatch for {city}. "
            f"Matched {len(comparison)} of {len(test_data)}."
        )

    differences = np.abs(
        comparison["predicted_price"]
        - comparison["regenerated_prediction"]
    )

    max_difference = differences.max()

    mean_difference = differences.mean()

    passed = np.allclose(
        comparison["predicted_price"],
        comparison["regenerated_prediction"],
        rtol=1e-7,
        atol=1e-8
    )

    results.append({

        "city": city,
        "test_records": len(comparison),
        "maximum_absolute_difference": max_difference,
        "mean_absolute_difference": mean_difference,
        "reproducibility_passed": passed

    })

    print(f"Test records: {len(comparison)}")
    print(f"Maximum difference: {max_difference:.12f}")
    print(f"Mean difference: {mean_difference:.12f}")
    print(f"Passed: {passed}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = (
    OUTPUT_DIR / "reproducibility_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n========== REPRODUCIBILITY SUMMARY ==========")

print(results_df.to_string(index=False))

print(
    f"\nOverall result: "
    f"{results_df['reproducibility_passed'].all()}"
)

print(f"\nResults saved to: {output_path}")