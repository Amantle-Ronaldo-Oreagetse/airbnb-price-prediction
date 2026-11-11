from pathlib import Path
import ast
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

LISTINGS_PATH = RAW_DIR / "Listings.csv"

CITY_DATA_DIR = PROCESSED_DIR / "city_datasets"

CITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA USING UTF-8
# ============================================================

print("\n========== LOADING LISTINGS ==========")

df = pd.read_csv(
    LISTINGS_PATH,
    encoding="utf-8",
    encoding_errors="replace"
)

print("Dataset shape:", df.shape)
print("Selected encoding: utf-8")


# ============================================================
# 3. INVESTIGATE POSSIBLE TEXT ENCODING CORRUPTION
# ============================================================

print("\n========== TEXT ENCODING INVESTIGATION ==========")

text_columns = [
    "name",
    "neighbourhood",
    "property_type"
]

for column in text_columns:

    print(f"\nColumn: {column}")

    samples = (
        df[column]
        .dropna()
        .astype(str)
    )

    # Display examples containing non-ASCII characters.
    non_ascii_samples = samples[
        samples.str.contains(r"[^\x00-\x7F]", regex=True)
    ]

    print("Non-ASCII examples:")

    for value in non_ascii_samples.head(10):
        print(" ", repr(value))

    # Count suspicious mojibake markers.
    suspicious_markers = ["Ã", "Â", "â€", "Å"]

    suspicious_count = samples.apply(
        lambda value: any(
            marker in value for marker in suspicious_markers
        )
    ).sum()

    print("Potentially corrupted values:", suspicious_count)


# ============================================================
# 4. VALIDATE PRIMARY KEY
# ============================================================

print("\n========== PRIMARY KEY VALIDATION ==========")

missing_ids = df["listing_id"].isna().sum()

duplicate_ids = df["listing_id"].duplicated().sum()

print("Missing listing IDs:", missing_ids)
print("Duplicate listing IDs:", duplicate_ids)

if missing_ids > 0 or duplicate_ids > 0:
    raise ValueError(
        "Listing ID validation failed. "
        "Resolve missing or duplicate IDs before proceeding."
    )


# ============================================================
# 5. INVESTIGATE AMENITIES FORMAT
# ============================================================

print("\n========== AMENITIES INVESTIGATION ==========")

amenities_samples = df["amenities"].dropna().head(10)

for value in amenities_samples:
    print(repr(value))


# ============================================================
# 6. EXTRACT AMENITIES COUNT
# ============================================================

def count_amenities(value):

    if pd.isna(value):
        return 0

    value = str(value).strip()

    if not value:
        return 0

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return len(parsed)

    except (ValueError, SyntaxError):
        pass

    # Fallback for comma-separated strings.
    cleaned = value.strip("[]{}")

    items = [
        item.strip().strip('"').strip("'")
        for item in cleaned.split(",")
        if item.strip()
    ]

    return len(items)


df["amenities_count"] = df["amenities"].apply(
    count_amenities
)


print("\nAmenities count summary:")

print(
    df["amenities_count"].describe().round(2)
)


# ============================================================
# 7. CATEGORICAL COMPLEXITY
# ============================================================

print("\n========== CATEGORICAL COMPLEXITY ==========")

complexity_rows = []

for city, city_df in df.groupby("city"):

    complexity_rows.append({

        "city": city,

        "listing_count": len(city_df),

        "unique_property_types":
            city_df["property_type"].nunique(),

        "unique_neighbourhoods":
            city_df["neighbourhood"].nunique(),

        "unique_room_types":
            city_df["room_type"].nunique(),

        "missing_property_type":
            city_df["property_type"].isna().sum(),

        "missing_neighbourhood":
            city_df["neighbourhood"].isna().sum()

    })


complexity_df = pd.DataFrame(complexity_rows)

print(complexity_df.to_string(index=False))

complexity_df.to_csv(
    REPORTS_DIR / "categorical_complexity_summary.csv",
    index=False
)


# ============================================================
# 8. PRICE VALIDATION
# ============================================================

print("\n========== PRICE VALIDATION ==========")

missing_prices = df["price"].isna().sum()

zero_prices = (df["price"] == 0).sum()

negative_prices = (df["price"] < 0).sum()

print("Missing prices:", missing_prices)
print("Zero prices:", zero_prices)
print("Negative prices:", negative_prices)

if missing_prices > 0:
    raise ValueError("Missing target prices detected.")

if negative_prices > 0:
    raise ValueError("Negative target prices detected.")


# Exclude zero-price listings from supervised modelling.
df = df.loc[df["price"] > 0].copy()

print("Listings retained:", len(df))


# ============================================================
# 9. MISSING BEDROOM INDICATOR
# ============================================================

df["bedrooms_missing"] = (
    df["bedrooms"].isna().astype(int)
)

print("\nBedroom missing indicator:")

print(
    df["bedrooms_missing"].value_counts()
)


# ============================================================
# 10. PREPARE CATEGORICAL FEATURES
# ============================================================

categorical_features = [

    "property_type",
    "room_type",
    "neighbourhood",
    "instant_bookable"

]

for column in categorical_features:

    df[column] = (
        df[column]
        .fillna("__MISSING__")
        .astype(str)
    )


# ============================================================
# 11. DEFINE MODEL FEATURES
# ============================================================

numeric_features = [

    "accommodates",
    "bedrooms",
    "bedrooms_missing",
    "minimum_nights",
    "maximum_nights",
    "amenities_count",
    "latitude",
    "longitude"

]


target = "price"

identifier_columns = [

    "listing_id",
    "host_id"

]


feature_definitions = {

    "target": target,

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "identifier_columns": identifier_columns,

    "excluded_from_model": [

        "listing_id",
        "host_id",
        "city",
        "name",
        "amenities",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value"

    ],

    "target_definition":
        "Advertised listing price in the city's local currency.",

    "prediction_scenario":
        "Estimate listing price using property characteristics "
        "and location information.",

    "currency_strategy":
        "Separate models for each city to avoid mixing currencies.",

    "zero_price_policy":
        "Exclude zero-price listings from supervised modelling.",

    "missing_bedrooms_policy":
        "Retain missingness indicator; impute using training data only."

}


with open(
    REPORTS_DIR / "feature_definitions.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        feature_definitions,
        file,
        indent=4
    )


# ============================================================
# 12. CREATE CITY-SPECIFIC TRAINING AND TESTING DATASETS
# ============================================================

print("\n========== CREATING CITY DATASETS ==========")

selected_columns = (
    identifier_columns
    + numeric_features
    + categorical_features
    + [target]
)

split_rows = []

for city, city_df in df.groupby("city"):

    city_name = (
        city.lower()
        .replace(" ", "_")
    )

    city_dir = CITY_DATA_DIR / city_name

    city_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    modelling_df = city_df[
        selected_columns
    ].copy()

    train_df, test_df = train_test_split(

        modelling_df,

        test_size=0.20,

        random_state=42

    )

    train_path = city_dir / "train.csv"
    test_path = city_dir / "test.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    split_rows.append({

        "city": city,

        "total_listings": len(modelling_df),

        "training_listings": len(train_df),

        "testing_listings": len(test_df)

    })

    print(
        f"{city}: "
        f"Total={len(modelling_df):,} | "
        f"Train={len(train_df):,} | "
        f"Test={len(test_df):,}"
    )


# ============================================================
# 13. SAVE SPLIT SUMMARY
# ============================================================

split_summary = pd.DataFrame(split_rows)

split_summary.to_csv(

    REPORTS_DIR / "city_train_test_summary.csv",

    index=False

)


# ============================================================
# 14. FINAL VALIDATION
# ============================================================

print("\n========== FINAL VALIDATION ==========")

total_processed = split_summary[
    "total_listings"
].sum()

total_training = split_summary[
    "training_listings"
].sum()

total_testing = split_summary[
    "testing_listings"
].sum()

print("Total processed listings:", total_processed)

print("Total training listings:", total_training)

print("Total testing listings:", total_testing)

print("Total cities:", len(split_summary))


assert total_processed == len(df)

assert total_training + total_testing == total_processed

assert total_processed == 279599, (
    "Unexpected processed listing count. "
    "Investigate before proceeding."
)

print("\nDATA PREPARATION COMPLETED SUCCESSFULLY.")