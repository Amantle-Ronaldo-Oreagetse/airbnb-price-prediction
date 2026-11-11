from pathlib import Path
import pandas as pd
import sqlite3

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\user\Documents\computational-materials\airbnb-price-prediction"
)

RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LISTINGS_FILE = RAW_DIR / "Listings.csv"
REVIEWS_FILE = RAW_DIR / "Reviews.csv"

LISTINGS_DICTIONARY = RAW_DIR / "Listings_data_dictionary.csv"
REVIEWS_DICTIONARY = RAW_DIR / "Reviews_data_dictionary.csv"

# Reviews are large, so process them in manageable chunks.
CHUNK_SIZE = 100_000


# ============================================================
# 2. VERIFY THE FOUR INPUT FILES
# ============================================================

files = {
    "Listings": LISTINGS_FILE,
    "Reviews": REVIEWS_FILE,
    "Listings dictionary": LISTINGS_DICTIONARY,
    "Reviews dictionary": REVIEWS_DICTIONARY
}

print("\n========== FILE INVENTORY ==========\n")

for name, path in files.items():

    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)

        print(f"{name}: FOUND ({size_mb:.2f} MB)")

    else:
        print(f"{name}: NOT FOUND - CHECK FILE NAME")


# ============================================================
# 3. INSPECT THE DATA DICTIONARIES
# ============================================================

print("\n========== DATA DICTIONARIES ==========\n")

for name, path in [
    ("Listings", LISTINGS_DICTIONARY),
    ("Reviews", REVIEWS_DICTIONARY)
]:

    if path.exists():

        dictionary = pd.read_csv(
            path,
            encoding="utf-8",
            encoding_errors="replace"
        )

        print(f"\n{name} dictionary:")
        print(dictionary.to_string(index=False))


# ============================================================
# 4. INSPECT LISTINGS DATASET
# ============================================================

print("\n========== LISTINGS INSPECTION ==========\n")

listings = pd.read_csv(
    LISTINGS_FILE,
    encoding="cp1252",
    encoding_errors="replace"
)

print("Dataset shape:", listings.shape)

print("\nColumn names:")
print(listings.columns.tolist())

print("\nData types:")
print(listings.dtypes)

print("\nFirst five records:")
print(listings.head())

print("\nDataset information:")
listings.info()


# ============================================================
# 5. PRIMARY KEY VALIDATION
# ============================================================

print("\n========== LISTINGS PRIMARY KEY ==========\n")

listing_id_missing = listings["listing_id"].isna().sum()

listing_id_duplicates = listings["listing_id"].duplicated().sum()

print("Missing listing IDs:", listing_id_missing)

print("Duplicate listing IDs:", listing_id_duplicates)

print(
    "Primary key valid:",
    listing_id_missing == 0 and listing_id_duplicates == 0
)


# ============================================================
# 6. MISSING VALUE ANALYSIS
# ============================================================

print("\n========== LISTINGS MISSING VALUES ==========\n")

missing = pd.DataFrame({
    "missing_count": listings.isna().sum(),
    "missing_percentage": (
        listings.isna().mean() * 100
    ).round(2)
})

missing = missing.sort_values(
    by="missing_percentage",
    ascending=False
)

print(missing)

missing.to_csv(
    REPORTS_DIR / "listings_missing_values.csv"
)


# ============================================================
# 7. DUPLICATE RECORD ANALYSIS
# ============================================================

print("\n========== LISTINGS DUPLICATES ==========\n")

print(
    "Completely duplicated rows:",
    listings.duplicated().sum()
)


# ============================================================
# 8. PRICE VALIDATION
# ============================================================

print("\n========== PRICE VALIDATION ==========\n")

listings["price"] = pd.to_numeric(
    listings["price"],
    errors="coerce"
)

print("Price summary:")
print(listings["price"].describe())

print(
    "Missing or unconvertible prices:",
    listings["price"].isna().sum()
)

print(
    "Zero or negative prices:",
    (listings["price"] <= 0).sum()
)

print(
    "Prices above 1,000:",
    (listings["price"] > 1000).sum()
)


# ============================================================
# 9. CITY DISTRIBUTION
# ============================================================

print("\n========== CITY DISTRIBUTION ==========\n")

city_counts = listings["city"].value_counts(dropna=False)

print(city_counts)

city_counts.to_csv(
    REPORTS_DIR / "listings_city_distribution.csv"
)


# ============================================================
# 10. LISTINGS CATEGORICAL VARIABLES
# ============================================================

print("\n========== CATEGORICAL VARIABLES ==========\n")

categorical_columns = [
    "city",
    "property_type",
    "room_type",
    "instant_bookable",
    "host_is_superhost"
]

for column in categorical_columns:

    if column in listings.columns:

        print(f"\n{column}:")

        print(
            listings[column]
            .value_counts(dropna=False)
            .head(20)
        )


# ============================================================
# 11. PREPARE LISTINGS IDS FOR RELATIONSHIP VALIDATION
# ============================================================

listing_ids = set(
    listings["listing_id"]
    .dropna()
    .astype(str)
)

del listings


# ============================================================
# 12. REVIEWS DATASET - CHUNKED INSPECTION
# ============================================================

print("\n========== REVIEWS INSPECTION ==========\n")

review_rows = 0

review_missing = {}

review_id_counts = {}

review_pair_counts = {}

unmatched_listing_count = 0

missing_listing_id_count = 0

missing_review_id_count = 0

first_chunk = True


for chunk in pd.read_csv(
    REVIEWS_FILE,
    chunksize=CHUNK_SIZE,
    encoding="utf-8",
    encoding_errors="replace",
    dtype={
        "listing_id": "string",
        "review_id": "string"
    }
):

    review_rows += len(chunk)

    # --------------------------------------------------------
    # First chunk: inspect structure
    # --------------------------------------------------------

    if first_chunk:

        print("First chunk shape:", chunk.shape)

        print("\nColumns:")
        print(chunk.columns.tolist())

        print("\nFirst five records:")
        print(chunk.head())

        print("\nData types:")
        print(chunk.dtypes)

        first_chunk = False

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column, count in chunk.isna().sum().items():

        review_missing[column] = (
            review_missing.get(column, 0) + count
        )

    # --------------------------------------------------------
    # Identifier validation
    # --------------------------------------------------------

    missing_listing_id_count += chunk["listing_id"].isna().sum()

    missing_review_id_count += chunk["review_id"].isna().sum()

    # --------------------------------------------------------
    # Listings-to-Reviews relationship
    # --------------------------------------------------------

    valid_listing_ids = chunk["listing_id"].dropna().astype(str)

    unmatched_listing_count += (
        ~valid_listing_ids.isin(listing_ids)
    ).sum()

    # --------------------------------------------------------
    # Review ID duplication
    # --------------------------------------------------------

    review_counts = (
        chunk["review_id"]
        .dropna()
        .value_counts()
    )

    for review_id, count in review_counts.items():

        review_id_counts[review_id] = (
            review_id_counts.get(review_id, 0) + count
        )

    # --------------------------------------------------------
    # Composite key duplication
    # listing_id + review_id
    # --------------------------------------------------------

    valid_pairs = chunk[
        chunk["listing_id"].notna()
        & chunk["review_id"].notna()
    ]

    pair_counts = (
        valid_pairs
        .groupby(["listing_id", "review_id"])
        .size()
    )

    for pair, count in pair_counts.items():

        review_pair_counts[pair] = (
            review_pair_counts.get(pair, 0) + count
        )


# ============================================================
# 13. REVIEWS AUDIT RESULTS
# ============================================================

print("\n========== REVIEWS AUDIT RESULTS ==========\n")

print("Total review records:", review_rows)

print("Missing listing IDs:", missing_listing_id_count)

print("Missing review IDs:", missing_review_id_count)

print(
    "Review listing IDs not found in Listings:",
    unmatched_listing_count
)

duplicate_review_ids = sum(
    count - 1
    for count in review_id_counts.values()
    if count > 1
)

distinct_repeated_review_ids = sum(
    count > 1
    for count in review_id_counts.values()
)

duplicate_composite_keys = sum(
    count - 1
    for count in review_pair_counts.values()
    if count > 1
)

print(
    "Repeated review ID occurrences:",
    duplicate_review_ids
)

print(
    "Distinct review IDs appearing more than once:",
    distinct_repeated_review_ids
)

print(
    "Repeated listing_id + review_id combinations:",
    duplicate_composite_keys
)


# ============================================================
# 14. SAVE REVIEWS MISSING VALUE REPORT
# ============================================================

reviews_missing_df = pd.DataFrame({
    "missing_count": review_missing
})

reviews_missing_df["missing_percentage"] = (
    reviews_missing_df["missing_count"] / review_rows * 100
).round(2)

reviews_missing_df = reviews_missing_df.sort_values(
    by="missing_percentage",
    ascending=False
)

print("\nReviews missing values:")
print(reviews_missing_df)

reviews_missing_df.to_csv(
    REPORTS_DIR / "reviews_missing_values.csv"
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n========== AUDIT COMPLETED ==========\n")

print("Listings records: 279,712 ")

print("Reviews records:", review_rows)

print("Reports saved in:", REPORTS_DIR)