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
# 3. INVESTIGATE ZERO ACCOMMODATION CAPACITY
# ============================================================

print("\n========== ZERO ACCOMMODATION CAPACITY ==========\n")

zero_accommodates = df[df["accommodates"] == 0]

print("Total records:", len(zero_accommodates))

print("\nDistribution by city:")

print(zero_accommodates["city"].value_counts())

print("\nDistribution by property type:")

print(zero_accommodates["property_type"].value_counts())

print("\nPrice summary:")

print(zero_accommodates["price"].describe())


# ============================================================
# 4. INVESTIGATE HIGH-PRICE LISTINGS
# ============================================================

print("\n========== HIGH-PRICE LISTING INVESTIGATION ==========\n")

high_price_records = df[
    df["price"] > 0
].sort_values(
    "price",
    ascending=False
)

columns = [
    "listing_id",
    "name",
    "city",
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "minimum_nights",
    "maximum_nights",
    "price"
]

print(high_price_records[columns].head(30).to_string(index=False))

high_price_records[columns].head(100).to_csv(
    REPORTS_DIR / "highest_100_price_records.csv",
    index=False
)


# ============================================================
# 5. INVESTIGATE PRICE DISTRIBUTION BY PROPERTY TYPE
# ============================================================

print("\n========== PROPERTY TYPE PRICE SUMMARY ==========\n")

valid_prices = df[df["price"] > 0]

property_summary = (
    valid_prices
    .groupby(["city", "room_type"])["price"]
    .agg(
        listing_count="count",
        median_price="median",
        mean_price="mean",
        maximum_price="max"
    )
)

print(property_summary.round(2).to_string())

property_summary.to_csv(
    REPORTS_DIR / "city_room_type_price_summary.csv"
)


price_summary = (
    valid_prices.groupby(["city", "room_type"])["price"]
    .agg(
        listing_count="count",
        median_price="median",
        mean_price="mean",
        percentile_95=lambda x: x.quantile(0.95),
        percentile_99=lambda x: x.quantile(0.99),
        maximum_price="max"
    )
)

print("\n========== EXTENDED PRICE SUMMARY ==========")
print(price_summary.round(2).to_string())

price_summary.to_csv(
    REPORTS_DIR / "extended_city_room_type_price_summary.csv"
)


# Identify observations above the 99th percentile
def identify_extreme_prices(group):
    threshold = group["price"].quantile(0.99)

    extreme = group[group["price"] > threshold]

    return pd.Series({
        "p99_threshold": threshold,
        "extreme_listing_count": len(extreme),
        "maximum_price": extreme["price"].max(),
        "median_extreme_price": extreme["price"].median()
    })


extreme_summary = (
    valid_prices.groupby(["city", "room_type"])
    .apply(identify_extreme_prices)
)

print("\n========== EXTREME PRICE RECORDS ==========")
print(extreme_summary.round(2).to_string())

extreme_summary.to_csv(
    REPORTS_DIR / "extreme_price_summary.csv")


# TOP RECORDS
top_records = (
    valid_prices
    .sort_values(
        ["city", "price"],
        ascending=[True, False]
    )
    .groupby("city")
    .head(50)
)

columns = [
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

top_records[columns].to_csv(
    REPORTS_DIR / "high_price_records_by_city.csv",
    index=False
)

print("Records extracted:", len(top_records))
print("\nRecords per city:")
print(top_records["city"].value_counts().to_string())


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n========== INVESTIGATION COMPLETED ==========\n")

print("Zero-accommodation records:", len(zero_accommodates))

print("Highest-price records exported:", min(100, len(high_price_records)))

print("Reports saved in:", REPORTS_DIR)