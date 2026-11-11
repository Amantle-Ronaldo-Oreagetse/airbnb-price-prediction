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

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit
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

PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "city_datasets"

REPORT_DIR = PROJECT_DIR / "reports" / "model_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

GROUP_COLUMN = "host_id"


# ============================================================
# 3. PREPROCESSING PIPELINE
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

    preprocessor = ColumnTransformer([
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

    return preprocessor


# ============================================================
# 4. MODEL CONFIGURATIONS
# ============================================================

def get_model_configurations():

    return {

        "Linear Regression": [
            {
                "model": LinearRegression(),
                "target": "original"
            }
        ],

        "Random Forest": [

            {
                "model": RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=42,
                    n_jobs=-1
                ),
                "target": "original"
            },

            {
                "model": RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                "target": "original"
            }
        ],

        "Gradient Boosting": [

            {
                "model": GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ),
                "target": "original"
            },

            {
                "model": GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ),
                "target": "original"
            }
        ],

        "HistGradientBoosting": [

            {
                "model": HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    random_state=42
                ),
                "target": "original"
            },

            {
                "model": HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42
                ),
                "target": "original"
            }
        ]
    }


# ============================================================
# 5. BUILD COMPLETE MODEL PIPELINE
# ============================================================

def build_model(model, target_type):

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "regressor",
            model
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
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    # Prices cannot be negative.
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(y, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 7. VALIDATION FOR ONE CITY
# ============================================================

def validate_city(city_dir):

    city = city_dir.name

    print("\n" + "=" * 65)
    print(f"VALIDATING CITY: {city}")
    print("=" * 65)

    train_path = city_dir / "train.csv"

    data = pd.read_csv(train_path)

    # Check required columns.
    required_columns = FEATURES + [TARGET, GROUP_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{city}: Missing required columns: {missing_columns}"
        )

    # Remove rows with invalid targets, if any.
    data = data.loc[
        data[TARGET].notna() &
        (data[TARGET] > 0)
    ].copy()

    X = data[FEATURES]
    y = data[TARGET]

    # --------------------------------------------------------
    # RANDOM TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    results = []

    # --------------------------------------------------------
    # MEDIAN BASELINE
    # --------------------------------------------------------

    baseline = DummyRegressor(strategy="median")

    baseline.fit(X_train, y_train)

    baseline_metrics = evaluate_model(
        baseline,
        X_val,
        y_val
    )

    results.append({

        "city": city,
        "model": "Median Baseline",
        "configuration": "Default",
        "target_transform": "original",
        **baseline_metrics

    })

    print(
        f"Median Baseline | "
        f"MAE: {baseline_metrics['MAE']:.3f}"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    configurations = get_model_configurations()

    for model_name, model_configs in configurations.items():

        for config_number, config in enumerate(
            model_configs,
            start=1
        ):

            for target_type in ["original", "log"]:

                model = build_model(
                    config["model"],
                    target_type
                )

                model.fit(
                    X_train,
                    y_train
                )

                metrics = evaluate_model(
                    model,
                    X_val,
                    y_val
                )

                configuration_name = (
                    f"Configuration {config_number}"
                )

                result = {

                    "city": city,
                    "model": model_name,
                    "configuration": configuration_name,
                    "target_transform": target_type,
                    **metrics

                }

                results.append(result)

                print(
                    f"{model_name} | "
                    f"{configuration_name} | "
                    f"{target_type} | "
                    f"MAE: {metrics['MAE']:.3f} | "
                    f"RMSE: {metrics['RMSE']:.3f} | "
                    f"R2: {metrics['R2']:.3f}"
                )

    # --------------------------------------------------------
    # SELECT BEST CONFIGURATION USING VALIDATION MAE
    # --------------------------------------------------------

    city_results = pd.DataFrame(results)

    city_results = city_results.sort_values(
        by=["MAE", "RMSE"],
        ascending=True
    )

    best_result = city_results.iloc[0]

    print("\nBEST VALIDATION CONFIGURATION")
    print(best_result)

    # --------------------------------------------------------
    # HOST-GROUPED VALIDATION
    # --------------------------------------------------------

    print("\nHOST-GROUPED VALIDATION")

    grouped_data = data.loc[
        data[GROUP_COLUMN].notna()
    ].copy()

    if grouped_data[GROUP_COLUMN].nunique() < 2:

        print(
            "Skipped: insufficient distinct host groups."
        )

        group_result = {
            "city": city,
            "status": "Skipped",
            "reason": "Insufficient distinct host groups"
        }

    else:

        X_group = grouped_data[FEATURES]
        y_group = grouped_data[TARGET]
        groups = grouped_data[GROUP_COLUMN]

        group_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42
        )

        group_train_idx, group_val_idx = next(
            group_splitter.split(
                X_group,
                y_group,
                groups=groups
            )
        )

        X_group_train = X_group.iloc[group_train_idx]
        X_group_val = X_group.iloc[group_val_idx]

        y_group_train = y_group.iloc[group_train_idx]
        y_group_val = y_group.iloc[group_val_idx]

        selected_model_name = best_result["model"]

        selected_config_number = int(
            best_result["configuration"].split()[-1]
        ) if best_result["model"] != "Median Baseline" else 0

        selected_target = best_result["target_transform"]

        if selected_model_name == "Median Baseline":

            group_model = DummyRegressor(
                strategy="median"
            )

        else:

            selected_config = configurations[
                selected_model_name
            ][selected_config_number - 1]

            group_model = build_model(
                selected_config["model"],
                selected_target
            )

        group_model.fit(
            X_group_train,
            y_group_train
        )

        group_metrics = evaluate_model(
            group_model,
            X_group_val,
            y_group_val
        )

        group_result = {

            "city": city,
            "status": "Completed",
            "selected_model": selected_model_name,
            "configuration": best_result["configuration"],
            "target_transform": selected_target,
            "training_rows": len(X_group_train),
            "validation_rows": len(X_group_val),
            "training_hosts": groups.iloc[group_train_idx].nunique(),
            "validation_hosts": groups.iloc[group_val_idx].nunique(),
            **group_metrics

        }

        print(
            f"Grouped MAE: {group_metrics['MAE']:.3f}"
        )

        print(
            f"Grouped RMSE: {group_metrics['RMSE']:.3f}"
        )

        print(
            f"Grouped R2: {group_metrics['R2']:.3f}"
        )

    return city_results, best_result, group_result


# ============================================================
# 8. RUN VALIDATION FOR ALL CITIES
# ============================================================

def main():

    all_results = []
    selected_models = []
    grouped_results = []

    city_directories = sorted([
        path
        for path in PROCESSED_DIR.iterdir()
        if path.is_dir()
    ])

    print(f"Cities discovered: {len(city_directories)}")

    for city_dir in city_directories:

        if not (city_dir / "train.csv").exists():

            print(
                f"Skipping {city_dir.name}: train.csv not found."
            )

            continue

        city_results, best_result, group_result = validate_city(
            city_dir
        )

        all_results.append(city_results)

        selected_models.append(
            best_result.to_dict()
        )

        grouped_results.append(group_result)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    all_results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    selected_models_df = pd.DataFrame(
        selected_models
    )

    grouped_results_df = pd.DataFrame(
        grouped_results
    )

    all_results_df.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False
    )

    selected_models_df.to_csv(
        REPORT_DIR / "selected_models_by_city.csv",
        index=False
    )

    grouped_results_df.to_csv(
        REPORT_DIR / "group_validation_results.csv",
        index=False
    )

    selected_models_df.to_json(
        REPORT_DIR / "selected_models_by_city.json",
        orient="records",
        indent=4
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STAGE 4.2 COMPLETE")
    print("=" * 65)

    print("\nSELECTED VALIDATION MODELS")

    print(
        selected_models_df[
            [
                "city",
                "model",
                "configuration",
                "target_transform",
                "MAE",
                "RMSE",
                "R2"
            ]
        ].to_string(index=False)
    )

    print("\nGROUPED VALIDATION SUMMARY")

    print(
        grouped_results_df.to_string(index=False)
    )

    print("\nFiles saved in:")
    print(REPORT_DIR)


if __name__ == "__main__":

    main()