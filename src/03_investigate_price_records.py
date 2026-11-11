from pathlib import Path
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\user\Documents\computational-materials\airbnb-price-prediction"
)

RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"

LISTINGS_FILE = RAW_DIR / "Listings.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    LISTINGS_FILE,
    encoding="cp1252",
    encoding_errors="replace"
)


# ============================================================
# 3. INVESTIGATE ZERO-PRICE LISTINGS
# ============================================================

print("\n========== ZERO-PRICE LISTINGS ==========\n")

zero_prices = df[df["price"] == 0][
    [
        "listing_id",
        "city",
        "property_type",
        "room_type",
        "accommodates",
        "bedrooms",
        "minimum_nights",
        "maximum_nights",
        "price"
    ]
]

print("Total zero-price listings:", len(zero_prices))

print(zero_prices.to_string(index=False))

zero_prices.to_csv(
    REPORTS_DIR / "zero_price_records.csv",
    index=False
)


# ============================================================
# 4. INVESTIGATE EXTREME PRICES BY CITY
# ============================================================

print("\n========== TOP 5 PRICES PER CITY ==========\n")

top_prices = (
    df.sort_values(
        ["city", "price"],
        ascending=[True, False]
    )
    .groupby("city")
    .head(5)
)

columns = [
    "listing_id",
    "city",
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "price"
]

print(top_prices[columns].to_string(index=False))

top_prices[columns].to_csv(
    REPORTS_DIR / "top_prices_by_city.csv",
    index=False
)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n========== INVESTIGATION COMPLETED ==========\n")

print("Zero-price records:", len(zero_prices))

print("Extreme-price records inspected:", len(top_prices))

print("Reports saved in:", REPORTS_DIR)