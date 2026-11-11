from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_DIR / "data" / "processed" / "city_datasets"
)

VALIDATION_DIR = (
    PROJECT_DIR / "reports" / "model_validation"
)

RESULTS_DIR = (
    PROJECT_DIR / "reports" / "final_test"
)

MODEL_DIR = (
    PROJECT_DIR / "models" / "final_models"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "bedrooms_missing",
    "minimum_nights",
    "maximum_nights",
    "amenities_count",
    "latitude",
    "longitude"
]

CATEGORICAL_FEATURES = [
    "property_type",
    "room_type",
    "neighbourhood",
    "instant_bookable"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "price"

IDENTIFIER_COLUMNS = [
    "listing_id",
    "host_id"
]


# ============================================================
# 3. LOAD SELECTED CONFIGURATIONS
# ============================================================

CONFIGURATION_PATH = (
    VALIDATION_DIR / "selected_models_by_city.json"
)

with open(CONFIGURATION_PATH, "r", encoding="utf-8") as file:

    selected_models = json.load(file)


selected_models_by_city = {
    item["city"]: item
    for item in selected_models
}


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=False,
                dtype=np.float32
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


# ============================================================
# 5. RECREATE MODEL CONFIGURATIONS
# ============================================================

def get_model_configurations():

    return {

        "Linear Regression": [
            LinearRegression()
        ],

        "Random Forest": [

            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=1.0,
                random_state=42,
                n_jobs=-1
            ),

            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )
        ],

        "Gradient Boosting": [

            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),

            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        ],

        "HistGradientBoosting": [

            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42
            ),

            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )
        ]
    }


# ============================================================
# 6. BUILD SELECTED MODEL
# ============================================================

def build_selected_model(model_name, configuration, target_type):

    configurations = get_model_configurations()

    config_number = int(
        configuration.split()[-1]
    )

    selected_estimator = configurations[
        model_name
    ][config_number - 1]

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "regressor",
            selected_estimator
        )
    ])

    if target_type == "log":

        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False
        )

    return pipeline


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(y_true, predictions):

    predictions = np.asarray(predictions)

    # Prevent negative predicted listing prices.
    clipped_count = int(
        np.sum(predictions < 0)
    )

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "negative_predictions_clipped": clipped_count,
        "predictions": predictions
    }


# ============================================================
# 8. TRAIN AND TEST ONE CITY
# ============================================================

def train_and_test_city(city_dir):

    city = city_dir.name

    print("\n" + "=" * 65)
    print(f"FINAL TRAINING AND TESTING: {city}")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_path = city_dir / "train.csv"
    test_path = city_dir / "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Training records: {len(train_data)}")
    print(f"Testing records:  {len(test_data)}")

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = FEATURES + [TARGET]

    for column in required_columns:

        if column not in train_data.columns:

            raise ValueError(
                f"{city}: Missing {column} in training data."
            )

        if column not in test_data.columns:

            raise ValueError(
                f"{city}: Missing {column} in testing data."
            )

    # --------------------------------------------------------
    # PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # --------------------------------------------------------
    # LOAD SELECTED MODEL
    # --------------------------------------------------------

    if city not in selected_models_by_city:

        raise ValueError(
            f"No selected configuration found for {city}."
        )

    selection = selected_models_by_city[city]

    model_name = selection["model"]

    configuration = selection["configuration"]

    target_type = selection["target_transform"]

    print(f"Selected model: {model_name}")
    print(f"Configuration:  {configuration}")
    print(f"Target:         {target_type}")

    # --------------------------------------------------------
    # TRAIN SELECTED MODEL
    # --------------------------------------------------------

    model = build_selected_model(
        model_name,
        configuration,
        target_type
    )

    model.fit(
        X_train,
        y_train
    )

    print("Final model training completed.")

    # --------------------------------------------------------
    # PREDICT TEST DATA
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    model_metrics = evaluate_predictions(
        y_test,
        predictions
    )

    print("\nSELECTED MODEL TEST PERFORMANCE")

    print(f"MAE:  {model_metrics['MAE']:.3f}")
    print(f"RMSE: {model_metrics['RMSE']:.3f}")
    print(f"R2:   {model_metrics['R2']:.3f}")

    # --------------------------------------------------------
    # MEDIAN BASELINE
    # --------------------------------------------------------

    baseline = DummyRegressor(
        strategy="median"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_predictions = baseline.predict(
        X_test
    )

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions
    )

    print("\nMEDIAN BASELINE TEST PERFORMANCE")

    print(f"MAE:  {baseline_metrics['MAE']:.3f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.3f}")
    print(f"R2:   {baseline_metrics['R2']:.3f}")

    # --------------------------------------------------------
    # PREDICTION IMPROVEMENT OVER BASELINE
    # --------------------------------------------------------

    improvement = (
        (
            baseline_metrics["MAE"] -
            model_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
    ) * 100

    print(
        f"\nMAE improvement over baseline: "
        f"{improvement:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_columns = {}

    for column in IDENTIFIER_COLUMNS:

        if column in test_data.columns:

            prediction_columns[column] = (
                test_data[column].values
            )

    prediction_columns["actual_price"] = (
        y_test.values
    )

    prediction_columns["predicted_price"] = (
        model_metrics["predictions"]
    )

    prediction_columns["prediction_error"] = (
        y_test.values -
        model_metrics["predictions"]
    )

    prediction_columns["absolute_error"] = np.abs(
        prediction_columns["prediction_error"]
    )

    predictions_df = pd.DataFrame(
        prediction_columns
    )

    predictions_df.to_csv(
        RESULTS_DIR / f"{city}_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------------

    model_package = {

        "model": model,

        "city": city,

        "model_name": model_name,

        "configuration": configuration,

        "target_transform": target_type,

        "features": FEATURES,

        "numeric_features": NUMERIC_FEATURES,

        "categorical_features": CATEGORICAL_FEATURES,

        "target": TARGET

    }

    joblib.dump(
        model_package,
        MODEL_DIR / f"{city}_final_model.joblib"
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "city": city,

        "model": model_name,

        "configuration": configuration,

        "target_transform": target_type,

        "training_rows": len(train_data),

        "testing_rows": len(test_data),

        "test_MAE": model_metrics["MAE"],

        "test_RMSE": model_metrics["RMSE"],

        "test_R2": model_metrics["R2"],

        "baseline_MAE": baseline_metrics["MAE"],

        "baseline_RMSE": baseline_metrics["RMSE"],

        "baseline_R2": baseline_metrics["R2"],

        "MAE_improvement_percent": improvement,

        "negative_predictions_clipped": (
            model_metrics["negative_predictions_clipped"]
        )

    }


# ============================================================
# 9. RUN ALL CITIES
# ============================================================

def main():

    results = []

    city_directories = sorted([
        path
        for path in PROCESSED_DIR.iterdir()
        if path.is_dir()
    ])

    print(f"Cities discovered: {len(city_directories)}")

    for city_dir in city_directories:

        if not (
            (city_dir / "train.csv").exists()
            and
            (city_dir / "test.csv").exists()
        ):

            print(
                f"Skipping {city_dir.name}: "
                "training or testing file missing."
            )

            continue

        city_result = train_and_test_city(
            city_dir
        )

        results.append(city_result)

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "final_test_model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STAGE 4.3 COMPLETE")
    print("=" * 75)

    summary_columns = [
        "city",
        "model",
        "test_MAE",
        "test_RMSE",
        "test_R2",
        "baseline_MAE",
        "MAE_improvement_percent"
    ]

    print(
        results_df[summary_columns].to_string(
            index=False
        )
    )

    print("\nFinal evaluation saved in:")
    print(RESULTS_DIR)

    print("\nFinal models saved in:")
    print(MODEL_DIR)


if __name__ == "__main__":

    main()