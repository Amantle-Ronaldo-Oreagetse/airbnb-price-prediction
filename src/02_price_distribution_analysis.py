from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\user\Documents\computational-materials\airbnb-price-prediction"
)

RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LISTINGS_FILE = RAW_DIR / "Listings.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

listings = pd.read_csv(
    LISTINGS_FILE,
    encoding="cp1252",
    encoding_errors="replace"
)

print("\n========== DATASET OVERVIEW ==========\n")

print("Dataset shape:", listings.shape)

print("\nCities:")
print(listings["city"].value_counts())


# ============================================================
# 3. PRICE VALIDATION
# ============================================================

listings["price"] = pd.to_numeric(
    listings["price"],
    errors="coerce"
)

print("\n========== PRICE VALIDATION ==========\n")

print("Missing prices:", listings["price"].isna().sum())

print("Zero prices:", (listings["price"] == 0).sum())

print("Negative prices:", (listings["price"] < 0).sum())


# ============================================================
# 4. CITY-SPECIFIC PRICE STATISTICS
# ============================================================

positive_prices = listings[
    listings["price"] > 0
].copy()

city_summary = (
    positive_prices
    .groupby("city")["price"]
    .agg(
        listing_count="count",
        minimum="min",
        mean="mean",
        median="median",
        standard_deviation="std",
        maximum="max"
    )
)

city_quantiles = (
    positive_prices
    .groupby("city")["price"]
    .quantile([0.25, 0.50, 0.75, 0.95, 0.99])
    .unstack()
)

city_quantiles.columns = [
    "25th_percentile",
    "50th_percentile",
    "75th_percentile",
    "95th_percentile",
    "99th_percentile"
]

city_summary = city_summary.join(city_quantiles)

print("\n========== CITY PRICE SUMMARY ==========\n")

print(city_summary.round(2).to_string())

city_summary.to_csv(
    REPORTS_DIR / "city_price_summary.csv"
)


# ============================================================
# 5. CITY-SPECIFIC IQR OUTLIER INVESTIGATION
# ============================================================

outlier_results = []

for city, group in positive_prices.groupby("city"):

    q1 = group["price"].quantile(0.25)

    q3 = group["price"].quantile(0.75)

    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr

    upper_fence = q3 + 1.5 * iqr

    outliers = group[
        (group["price"] < lower_fence)
        | (group["price"] > upper_fence)
    ]

    high_outliers = group[
        group["price"] > upper_fence
    ]

    outlier_results.append({
        "city": city,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "total_outliers": len(outliers),
        "high_price_outliers": len(high_outliers),
        "outlier_percentage": (
            len(outliers) / len(group) * 100
        )
    })


outlier_summary = pd.DataFrame(outlier_results)

outlier_summary = outlier_summary.sort_values(
    "outlier_percentage",
    ascending=False
)

print("\n========== CITY-SPECIFIC OUTLIER ANALYSIS ==========\n")

print(outlier_summary.round(2).to_string(index=False))

outlier_summary.to_csv(
    REPORTS_DIR / "city_price_outliers.csv",
    index=False
)


# ============================================================
# 6. ZERO-PRICE LISTINGS BY CITY
# ============================================================

zero_price_summary = (
    listings.assign(
        zero_price=listings["price"] == 0
    )
    .groupby("city")["zero_price"]
    .agg(
        zero_price_count="sum",
        total_listings="count"
    )
)

zero_price_summary["zero_price_percentage"] = (
    zero_price_summary["zero_price_count"]
    / zero_price_summary["total_listings"]
    * 100
)

print("\n========== ZERO-PRICE LISTINGS ==========\n")

print(zero_price_summary.round(3).to_string())

zero_price_summary.to_csv(
    REPORTS_DIR / "zero_price_by_city.csv"
)


# ============================================================
# 7. PRICE DISTRIBUTION VISUALISATION
# ============================================================

plot_data = positive_prices.copy()

plt.figure(figsize=(13, 7))

plot_data.boxplot(
    column="price",
    by="city",
    rot=45,
    showfliers=False
)

plt.yscale("log")

plt.title("Airbnb Listing Price Distribution by City")

plt.suptitle("")

plt.xlabel("City")

plt.ylabel("Listing Price (log scale)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "price_distribution_by_city.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n========== ANALYSIS COMPLETED ==========\n")

print("Reports saved in:", REPORTS_DIR)

print("Figures saved in:", FIGURES_DIR)