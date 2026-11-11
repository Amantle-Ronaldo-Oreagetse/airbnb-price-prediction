# Airbnb Listing Price Prediction Model

## 1. Project Overview

This project develops machine learning models to estimate Airbnb listing prices across ten international cities.

The project was undertaken as part of the AnalystLab Africa Machine Learning Internship (Track C).

The objective is to investigate how listing characteristics, accommodation capacity, amenities and geographical location relate to advertised prices, and to develop predictive models that estimate listing prices.

Because prices are recorded in different local currencies, ten independent city-specific models were developed rather than combining all listings into one model.

### Cities Covered

- Bangkok, Thailand
- Cape Town, South Africa
- Hong Kong
- Istanbul, Türkiye
- Mexico City, Mexico
- New York, United States
- Paris, France
- Rio de Janeiro, Brazil
- Rome, Italy
- Sydney, Australia

## 2. Business Problem

Airbnb hosts need to determine competitive listing prices. Setting prices too high may reduce booking interest, while setting them too low may reduce potential earnings.

This project investigates whether machine learning can estimate advertised listing prices from available property characteristics and geographical information.

The prediction scenario assumes that an estimate is generated when a host is creating or initially pricing a listing.

This is a modelling assumption because the dataset does not explicitly establish the exact point at which a prediction would be generated.

## 3. Dataset

The project uses two datasets supplied with the assignment:

- `Listings.csv`: 279,712 Airbnb listing records containing 33 columns.
- `Reviews.csv`: 5,373,143 review records containing listing identifiers, review identifiers, dates and reviewer identifiers.

Two associated data dictionaries were also supplied.

The original datasets are retained in `data/raw/` and are not modified by the preprocessing workflow.

### Data Quality Findings

The initial audit established that:

- All 279,712 listing records had unique listing IDs.
- The Reviews dataset contained 5,373,143 records.
- Every review listing ID matched an existing listing ID.
- Repeated review IDs existed, but the combination of `listing_id` and `review_id` was unique.
- There were 113 listings with zero advertised prices.
- Several listing attributes contained substantial missingness.
- Price distributions were strongly right-skewed across cities.

The 113 zero-price records were excluded from supervised model training. They were retained in the original raw dataset.

The Reviews dataset was audited to validate its structure and relationship with Listings. However, it was not used to construct predictive features because it contained review metadata rather than review text or numerical review ratings.

### Currency Considerations

Prices are recorded in each country's local currency. The dataset does not contain a dedicated currency column.

Separate city-specific models were therefore developed.

| City | Currency |
|---|---|
| Bangkok | THB |
| Cape Town | ZAR |
| Hong Kong | HKD |
| Istanbul | TRY |
| Mexico City | MXN |
| New York | USD |
| Paris | EUR |
| Rio de Janeiro | BRL |
| Rome | EUR |
| Sydney | AUD |

Predictions must be interpreted using the currency associated with the selected city. Values should not be compared directly across cities without currency conversion.

## 4. Methodology

The project followed a structured machine learning workflow.

### Stage 1: Data Understanding and Quality Audit

- Inspected dataset structures and data dictionaries.
- Checked missing values and duplicate records.
- Validated primary-key uniqueness.
- Examined relationships between listings and reviews.
- Investigated invalid and suspicious price records.

### Stage 2: Exploratory Data Analysis

- Examined city-level price distributions.
- Calculated descriptive statistics and percentiles.
- Investigated zero-price listings.
- Examined extreme prices and potential outliers.
- Compared price distributions across cities.

### Stage 3: Data Preparation and Feature Engineering

The target variable was `price`.

Selected numerical features:

- `accommodates`
- `bedrooms`
- `bedrooms_missing`
- `minimum_nights`
- `maximum_nights`
- `amenities_count`
- `latitude`
- `longitude`

Selected categorical features:

- `property_type`
- `room_type`
- `neighbourhood`
- `instant_bookable`

Additional preparation included:

- Creating an amenities count from the amenities field.
- Creating an indicator for missing bedroom information.
- Excluding zero-price records from supervised modelling.
- Handling missing values through preprocessing pipelines.
- Applying one-hot encoding to categorical variables.
- Splitting each city's data into training and testing sets.

The resulting modelling dataset contained 279,599 records after excluding the 113 zero-price records.

Each city's data was divided into approximately 80% training data and 20% independent test data.

### Stage 4: Model Development and Evaluation

The following approaches were investigated:

- Median baseline
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- Histogram-based Gradient Boosting Regression

Original and logarithmically transformed target values were evaluated.

Candidate models and configurations were compared using a validation split derived from the training data. Validation MAE was the primary selection metric, with RMSE used as a tiebreaker.

Additional grouped validation by host was conducted to investigate generalisation to previously unseen hosts. This was an additional diagnostic rather than the primary model-selection procedure.

The selected configurations were subsequently trained on the full training datasets and evaluated on the untouched test datasets.

### Stage 5: Model Packaging and Reproducibility

The final city-specific models were saved using Joblib.

A prediction interface was developed to:

- Load the appropriate model based on the selected city.
- Validate input features.
- Generate price predictions.
- Return the corresponding local currency.

Automated tests were implemented to check model loading, input validation and prediction functionality.

All ten saved models also passed reproducibility testing against their original final-test predictions.

## 5. Final Model Performance

The final models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

| City | Selected Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Bangkok | Random Forest | 1,077.17 | 7,742.07 | 0.053 |
| Cape Town | Random Forest | 992.73 | 4,017.22 | 0.440 |
| Hong Kong | Random Forest | 380.70 | 2,535.77 | 0.046 |
| Istanbul | Random Forest | 392.81 | 4,840.00 | 0.006 |
| Mexico City | Random Forest | 535.56 | 3,326.85 | 0.088 |
| New York | Random Forest | 51.97 | 214.02 | 0.212 |
| Paris | Histogram-based Gradient Boosting | 39.66 | 203.92 | 0.129 |
| Rio de Janeiro | Random Forest | 517.63 | 4,276.62 | 0.026 |
| Rome | Random Forest | 45.85 | 280.04 | 0.111 |
| Sydney | Histogram-based Gradient Boosting | 90.52 | 486.62 | 0.189 |

All ten models achieved lower MAE than their respective median baseline models.

Cape Town achieved the strongest test-set R² of approximately 0.440.

However, explanatory performance varied considerably across cities. Several models achieved relatively low R² values, indicating that the available features explain only a limited proportion of observed price variation in those markets.

MAE and RMSE values should not be directly compared across cities because prices are expressed in different currencies.

## 6. Model Interpretation and Error Analysis

Permutation feature importance was used to investigate the predictive contribution of individual features.

Frequently influential features included:

- Number of bedrooms
- Accommodation capacity
- Latitude and longitude
- Property type
- Room type

Their relative importance varied between cities.

Error analysis revealed several consistent patterns:

- All ten models showed positive mean residuals, indicating average underprediction.
- The models generally overestimated prices in the lowest price quartile.
- The models consistently underestimated prices in the highest price quartile.
- The highest price quartile had substantially greater absolute prediction errors.
- A small number of extreme price observations contributed disproportionately to RMSE in some cities.

These findings suggest that the models have difficulty estimating prices at the extremes of the observed price distributions.

Permutation importance describes predictive contribution, not causation.

## 7. Running the Project

### Requirements

- Python 3.10 or later, subject to compatibility with the installed dependencies.
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Joblib

The complete dependency lists are provided in `requirements.txt` and `requirements-lock.txt`.

### Installation

1. Clone the repository

git clone https://github.com/Amantle-Ronaldo-Oreagetse/airbnb-price-prediction.git
cd airbnb-price-prediction

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment (Windows PowerShell)

.\venv\Scripts\Activate.ps1

4. Install dependencies

python -m pip install -r requirements.txt

Dataset Setup

Place the assignment datasets and associated data dictionaries in:

data/raw/

The original datasets are not included in this repository unless redistribution has been authorised. They must be obtained separately.

Generate a Prediction

Run:

python src/10_predict_listing_price.py

The script demonstrates an example prediction using the Paris model.

The inference interface supports the available city-specific models. The example input and selected city can be modified within the script.

Run Automated Tests

python -m unittest discover -s tests -v

Verify Model Reproducibility

python src/11_reproducibility_check.py

This checks whether the saved models reproduce their original final-test predictions.


## 8. Project Structure

```text
airbnb-price-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── city_datasets/
│   └── external/
│
├── notebooks/
│   └── Airbnb_Price_Prediction.ipynb
│
├── src/
│   ├── 01_data_quality_audit.py
│   ├── 02_price_distribution_analysis.py
│   ├── 03_investigate_price_records.py
│   ├── 04_price_anomaly_investigation.py
│   ├── 05_prepare_modelling_data.py
│   ├── 06_train_and_evaluate_models.py
│   ├── 07_validate_and_tune_models.py
│   ├── 08_final_train_and_test.py
│   ├── 09_final_model_error_analysis.py
│   ├── 10_predict_listing_price.py
│   └── 11_reproducibility_check.py
│
├── models/
│   └── final_models/
│
├── reports/
│   ├── figures/
│   ├── model_validation/
│   ├── final_test/
│   ├── error_analysis/
│   └── inference_examples/
│
├── tests/
│   └── test_inference.py
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── .gitignore
```

Additional reports and figures are retained in their respective directories.

## 9. Limitations

This project has several limitations:

1. **Price distribution:** The data contains highly skewed price distributions and extreme observations, making accurate predictions at the upper end difficult.

2. **City-specific performance:** Predictive performance differs substantially across cities. Some markets have relatively weak explanatory performance.

3. **Limited features:** The available listing characteristics cannot capture every factor influencing accommodation prices.

4. **Temporal information:** The models do not account for seasonal demand, holidays, major events or changing market conditions.

5. **Prediction scenario:** The initial-pricing scenario is an assumption. Some available listing characteristics may not be known when a listing is first created.

6. **Currency:** Each prediction is expressed in the local currency of its selected city. Outputs are not directly comparable across cities.

7. **Model generalisation:** Grouped validation was conducted as an additional diagnostic, but performance under entirely new market conditions has not been established.

8. **Review information:** Although the Reviews dataset was audited, its available fields did not provide review text or numerical review ratings for feature engineering.

The models should therefore be treated as experimental price-estimation tools rather than definitive pricing recommendations.

## 10. Conclusion

This project demonstrates an end-to-end machine learning workflow involving data quality assessment, exploratory analysis, feature engineering, model comparison, validation, independent testing, error analysis and model packaging.

Ten city-specific models were developed and saved. Each outperformed its median baseline on MAE, although performance varied across markets.

The project also demonstrates the importance of reproducibility: all ten saved models successfully reproduced their original test predictions across 55,924 test records, with differences within floating-point tolerance.

Future improvements could investigate additional property and market features, alternative modelling approaches, improved treatment of extreme prices and more robust temporal validation.

## 11. Acknowledgements

This project was completed as part of the AnalystLab Africa Machine Learning Internship, Track C.

The datasets and associated data dictionaries were provided for the project assignment.