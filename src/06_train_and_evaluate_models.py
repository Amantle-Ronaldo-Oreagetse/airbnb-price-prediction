from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CITY_DATA_DIR = PROCESSED_DIR / "city_datasets"

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "reports" / "model_evaluation"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

IDENTIFIERS = [
    "listing_id",
    "host_id"
]


# ============================================================
# 3. DEFINE MODELS
# ============================================================

def get_models():

    return {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        )

    }


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():

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
# 5. CREATE COMPLETE MODEL PIPELINE
# ============================================================

def create_pipeline(model):

    return Pipeline([

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            model
        )

    ])


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_actual,
    y_predicted
):

    mae = mean_absolute_error(
        y_actual,
        y_predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_predicted
        )
    )

    r2 = r2_score(
        y_actual,
        y_predicted
    )

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    }


# ============================================================
# 7. LOAD CITY DATASETS
# ============================================================

city_folders = sorted([
    folder
    for folder in CITY_DATA_DIR.iterdir()
    if folder.is_dir()
])

if not city_folders:
    raise FileNotFoundError(
        "No city datasets found. "
        "Run the data preparation script first."
    )


all_results = []

city_best_models = {}


# ============================================================
# 8. TRAIN AND EVALUATE EACH CITY
# ============================================================

for city_folder in city_folders:

    city_name = city_folder.name

    print("\n" + "=" * 65)
    print(f"PROCESSING CITY: {city_name.upper()}")
    print("=" * 65)

    train_path = city_folder / "train.csv"
    test_path = city_folder / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))

    # --------------------------------------------------------
    # 8A. MEDIAN BASELINE
    # --------------------------------------------------------

    print("\nEvaluating median baseline...")

    baseline_prediction = np.full(
        len(y_test),
        y_train.median()
    )

    baseline_metrics = evaluate_model(
        y_test,
        baseline_prediction
    )

    city_results = []

    city_results.append({

        "city": city_name,

        "model": "Median Baseline",

        "target_transform": "Original",

        **baseline_metrics

    })

    print(
        f"Median Baseline | "
        f"MAE: {baseline_metrics['MAE']:.4f} | "
        f"RMSE: {baseline_metrics['RMSE']:.4f} | "
        f"R2: {baseline_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # 8B. ORIGINAL-TARGET MODELS
    # --------------------------------------------------------

    print("\nTraining original-target models...")

    models = get_models()

    fitted_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = create_pipeline(model)

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_test)

        metrics = evaluate_model(
            y_test,
            predictions
        )

        city_results.append({

            "city": city_name,

            "model": model_name,

            "target_transform": "Original",

            **metrics

        })

        fitted_models[model_name] = pipeline

        print(
            f"{model_name} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"R2: {metrics['R2']:.4f}"
        )

    # --------------------------------------------------------
    # 8C. SELECT TWO BEST ORIGINAL-TARGET MODELS
    # --------------------------------------------------------

    original_results = pd.DataFrame([
        result
        for result in city_results
        if result["target_transform"] == "Original"
        and result["model"] != "Median Baseline"
    ])

    top_two = (
        original_results
        .sort_values("MAE")
        .head(2)
    )

    print("\nBest two original-target models:")

    print(
        top_two[
            ["model", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 8D. LOG-TRANSFORMED TARGET MODELS
    # --------------------------------------------------------

    print("\nTesting log-transformed targets...")

    for model_name in top_two["model"]:

        print(f"\nLog-target training: {model_name}")

        base_model = get_models()[model_name]

        pipeline = create_pipeline(base_model)

        log_model = TransformedTargetRegressor(

            regressor=pipeline,

            func=np.log1p,

            inverse_func=np.expm1,

            check_inverse=False

        )

        log_model.fit(
            X_train,
            y_train
        )

        predictions = log_model.predict(X_test)

        metrics = evaluate_model(
            y_test,
            predictions
        )

        city_results.append({

            "city": city_name,

            "model": model_name,

            "target_transform": "Log",

            **metrics

        })

        print(
            f"{model_name} (Log) | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"R2: {metrics['R2']:.4f}"
        )

    # --------------------------------------------------------
    # 8E. IDENTIFY BEST MODEL FOR CITY
    # --------------------------------------------------------

    city_results_df = pd.DataFrame(city_results)

    best_result = (
        city_results_df
        .sort_values("MAE")
        .iloc[0]
    )

    best_model_name = best_result["model"]

    best_transform = best_result["target_transform"]

    print("\nBEST MODEL FOR", city_name.upper())

    print(best_result.to_string())

    # --------------------------------------------------------
    # 8F. RETRIEVE BEST MODEL
    # --------------------------------------------------------

    if best_transform == "Original":

        best_model = fitted_models[best_model_name]

    else:

        base_model = get_models()[best_model_name]

        pipeline = create_pipeline(base_model)

        best_model = TransformedTargetRegressor(

            regressor=pipeline,

            func=np.log1p,

            inverse_func=np.expm1,

            check_inverse=False

        )

        best_model.fit(
            X_train,
            y_train
        )

    # --------------------------------------------------------
    # 8G. SAVE BEST MODEL
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR /
        f"{city_name}_best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # --------------------------------------------------------
    # 8H. SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    best_predictions = best_model.predict(X_test)

    prediction_df = pd.DataFrame({

        "listing_id": test_df["listing_id"],

        "host_id": test_df["host_id"],

        "actual_price": y_test,

        "predicted_price": best_predictions,

        "prediction_error": (
            y_test.to_numpy() - best_predictions
        ),

        "absolute_error": np.abs(
            y_test.to_numpy() - best_predictions
        )

    })

    prediction_path = (
        RESULTS_DIR /
        f"{city_name}_best_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    city_best_models[city_name] = {

        "model": best_model_name,

        "target_transform": best_transform,

        "MAE": float(best_result["MAE"]),

        "RMSE": float(best_result["RMSE"]),

        "R2": float(best_result["R2"]),

        "model_file": model_path.name

    }

    all_results.extend(city_results)


# ============================================================
# 9. SAVE COMPLETE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    ["city", "MAE"]
)

results_df.to_csv(

    RESULTS_DIR / "model_comparison_all_cities.csv",

    index=False

)


# ============================================================
# 10. SAVE BEST MODEL SUMMARY
# ============================================================

best_models_df = pd.DataFrame.from_dict(
    city_best_models,
    orient="index"
)

best_models_df.index.name = "city"

best_models_df.to_csv(
    RESULTS_DIR / "best_models_by_city.csv"
)


with open(
    RESULTS_DIR / "best_models_by_city.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        city_best_models,
        file,
        indent=4
    )


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)

print("STAGE 4 COMPLETED")

print("=" * 65)

print("\nBEST MODEL PER CITY:")

print(best_models_df.to_string())

print("\nTOTAL MODEL EVALUATIONS:", len(results_df))

print("\nRESULT FILES SAVED TO:")
print(RESULTS_DIR)

print("\nSAVED MODELS:")
print(MODELS_DIR)