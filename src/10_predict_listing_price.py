from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models" / "final_models"
OUTPUT_DIR = PROJECT_DIR / "reports" / "inference_examples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CITY AND CURRENCY DEFINITIONS
# ============================================================

CITY_CURRENCIES = {
    "Bangkok": "THB",
    "Cape Town": "ZAR",
    "Hong Kong": "HKD",
    "Istanbul": "TRY",
    "Mexico City": "MXN",
    "New York": "USD",
    "Paris": "EUR",
    "Rio de Janeiro": "BRL",
    "Rome": "EUR",
    "Sydney": "AUD"
}


# ============================================================
# MODEL FEATURES
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

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_city_model(city):

    if city not in CITY_CURRENCIES:
        raise ValueError(
            f"Unsupported city: {city}. "
            f"Available cities: {list(CITY_CURRENCIES.keys())}"
        )

    model_path = MODEL_DIR / f"{city.lower().replace(' ', '_')}_final_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    package = joblib.load(model_path)

    # The Stage 4.3 package stores the fitted estimator
    # under the "model" key.
    if isinstance(package, dict):
        model = package["model"]
    else:
        model = package

    return model


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_listing_data(data):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = [
        column for column in MODEL_FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    data = data[MODEL_FEATURES].copy()

    if (data["accommodates"].dropna() < 0).any():
        raise ValueError("accommodates cannot be negative.")

    if (data["bedrooms"].dropna() < 0).any():
        raise ValueError("bedrooms cannot be negative.")

    if (data["minimum_nights"].dropna() < 0).any():
        raise ValueError("minimum_nights cannot be negative.")

    if (data["maximum_nights"].dropna() < 0).any():
        raise ValueError("maximum_nights cannot be negative.")

    if (data["amenities_count"].dropna() < 0).any():
        raise ValueError("amenities_count cannot be negative.")

    # Ensure the missing-bedroom indicator is correct.
    data["bedrooms_missing"] = (
        data["bedrooms"].isna().astype(int)
    )

    return data


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def predict_listing_price(city, listing_data):

    if city not in CITY_CURRENCIES:
        raise ValueError(f"Unsupported city: {city}")

    model = load_city_model(city)

    validated_data = validate_listing_data(listing_data)

    predictions = model.predict(validated_data)

    predictions = np.asarray(predictions).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError("The model produced invalid predictions.")

    # Prices cannot be negative.
    predictions = np.maximum(predictions, 0)

    results = pd.DataFrame({
        "city": city,
        "predicted_price": predictions,
        "currency": CITY_CURRENCIES[city]
    })

    return results


# ============================================================
# EXAMPLE PREDICTION
# ============================================================

if __name__ == "__main__":

    example_listing = pd.DataFrame([

        {
            "accommodates": 2,
            "bedrooms": 1,
            "bedrooms_missing": 0,
            "minimum_nights": 2,
            "maximum_nights": 30,
            "amenities_count": 20,
            "latitude": 48.8566,
            "longitude": 2.3522,
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "neighbourhood": "Louvre",
            "instant_bookable": "f"
        }

    ])

    prediction = predict_listing_price(
        city="Paris",
        listing_data=example_listing
    )

    print("\n========== LISTING PRICE PREDICTION ==========")

    print(prediction.to_string(index=False))

    output_path = OUTPUT_DIR / "example_prediction.csv"

    prediction.to_csv(output_path, index=False)

    print(f"\nPrediction saved to: {output_path}")