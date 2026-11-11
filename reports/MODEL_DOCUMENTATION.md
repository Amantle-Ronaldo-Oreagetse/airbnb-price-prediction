# Model Documentation

## 1. Model Purpose

The purpose of the Airbnb Listing Price Prediction Model is to estimate advertised listing prices using available property characteristics and geographical information.

Ten independent models were developed, one for each city.

The target variable is `price`, expressed in the local currency of the respective city.

The prediction scenario assumes that an estimate is generated when a host is creating or initially pricing a listing. This is a modelling assumption rather than a confirmed operational scenario.

## 2. Prediction Inputs

| Feature | Type | Description |
|---|---|---|
| accommodates | Numerical | Number of guests accommodated |
| bedrooms | Numerical | Number of bedrooms |
| bedrooms_missing | Numerical | Indicates missing bedroom information |
| minimum_nights | Numerical | Minimum permitted booking duration |
| maximum_nights | Numerical | Maximum permitted booking duration |
| amenities_count | Numerical | Number of listed amenities |
| latitude | Numerical | Geographic latitude |
| longitude | Numerical | Geographic longitude |
| property_type | Categorical | Property classification |
| room_type | Categorical | Type of accommodation |
| neighbourhood | Categorical | Listing neighbourhood |
| instant_bookable | Categorical | Instant-booking availability |

The listing ID and host ID are retained for traceability but excluded from model training.

## 3. Preprocessing

Numerical features are processed using median imputation and standardisation.

Categorical features are processed using most-frequent imputation and one-hot encoding.

The categorical encoder uses unknown-category handling and a minimum category frequency of ten.

Preprocessing is included in the fitted model pipelines to ensure consistent transformations during training and inference.

The same feature definitions and preprocessing pipeline are used when generating predictions.

## 4. Candidate Models and Selection Procedure

Candidate algorithms included:

- Median baseline
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- Histogram-based Gradient Boosting Regression

Both original and logarithmically transformed target values were evaluated.

Model selection was performed using validation data derived from the original training data.

Validation MAE was the primary selection metric, with RMSE used as a tiebreaker.

An additional grouped validation by host was performed to investigate performance on listings belonging to previously unseen hosts. This was a generalisation diagnostic and was not used as the primary model-selection procedure.

All ten selected configurations used logarithmically transformed targets.

The final selected configurations were subsequently retrained using the full training datasets and evaluated on independent test datasets that were not used for model selection.

## 5. Final Model Selection

| City | Algorithm | Configuration |
|---|---|---|
| Bangkok | Random Forest | Configuration 1 |
| Cape Town | Random Forest | Configuration 1 |
| Hong Kong | Random Forest | Configuration 1 |
| Istanbul | Random Forest | Configuration 1 |
| Mexico City | Random Forest | Configuration 1 |
| New York | Random Forest | Configuration 1 |
| Paris | Histogram-based Gradient Boosting | Configuration 2 |
| Rio de Janeiro | Random Forest | Configuration 1 |
| Rome | Random Forest | Configuration 2 |
| Sydney | Histogram-based Gradient Boosting | Configuration 2 |

### Random Forest Configuration 1

- `n_estimators=200`
- `min_samples_leaf=2`
- `max_features=1.0`
- `random_state=42`
- `n_jobs=-1`

### Random Forest Configuration 2

- `n_estimators=300`
- `min_samples_leaf=5`
- `max_features=0.8`
- `random_state=42`
- `n_jobs=-1`

### Histogram-based Gradient Boosting Configuration 2

- `max_iter=250`
- `learning_rate=0.05`
- `max_leaf_nodes=31`
- `l2_regularization=1.0`
- `random_state=42`

The selected model and configuration for each city are also recorded in:

- `reports/model_validation/selected_models_by_city.csv`
- `reports/model_validation/selected_models_by_city.json`

## 6. Independent Test Results

The final models were evaluated using the untouched test datasets.

| City | MAE | RMSE | R² |
|---|---:|---:|---:|
| Bangkok | 1,077.17 | 7,742.07 | 0.053 |
| Cape Town | 992.73 | 4,017.22 | 0.440 |
| Hong Kong | 380.70 | 2,535.77 | 0.046 |
| Istanbul | 392.81 | 4,840.00 | 0.006 |
| Mexico City | 535.56 | 3,326.85 | 0.088 |
| New York | 51.97 | 214.02 | 0.212 |
| Paris | 39.66 | 203.92 | 0.129 |
| Rio de Janeiro | 517.63 | 4,276.62 | 0.026 |
| Rome | 45.85 | 280.04 | 0.111 |
| Sydney | 90.52 | 486.62 | 0.189 |

All ten models achieved lower MAE than their respective median baselines.

Cape Town recorded the highest R² at approximately 0.440.

Istanbul, Rio de Janeiro, Bangkok and Hong Kong showed particularly limited explanatory performance.

MAE and RMSE are expressed in each city's local currency and should not be compared directly across cities.

## 7. Error Analysis

Residual analysis identified an overall tendency toward underprediction.

The models generally overestimated lower-priced listings and underestimated higher-priced listings.

The highest price quartile consistently recorded substantially greater absolute errors than the lowest price quartile.

The largest prediction errors were concentrated among unusually expensive listings.

This is consistent with the strongly right-skewed price distributions observed during exploratory data analysis.

The results suggest that the models have difficulty capturing the characteristics of the highest-priced properties.

A small number of extreme prediction errors also contributed disproportionately to RMSE in several cities.

## 8. Feature Importance

Permutation importance was used to estimate how much model performance deteriorated when individual features were shuffled.

Important features differed by city.

Bedrooms, accommodation capacity, geographical coordinates, room type and property type were frequently influential.

These results describe predictive importance and should not be interpreted as causal relationships.

Correlated features may also share or obscure predictive importance.

## 9. Validation and Reproducibility

The final models were trained on the full training datasets after model selection.

Independent test datasets were used for final performance evaluation.

Grouped validation by host was also conducted as an additional generalisation diagnostic. Listings with missing host identifiers were excluded from this grouped evaluation.

The ten saved model artifacts were subsequently checked against the original Stage 4.3 predictions.

### Reproducibility Results

- Models checked: 10
- Test records checked: 55,924
- Models passing reproducibility checks: 10
- Overall reproducibility result: PASS
- Maximum absolute prediction difference: approximately 1.31 × 10⁻¹⁰

The differences were within floating-point tolerance.

This confirms that the saved models reproduce their original predictions.

## 10. Deployment Considerations

The inference interface requires the user to select a city and provide the twelve model features.

The selected city determines which model is loaded and which local currency is returned.

The interface includes input validation and automated tests covering model loading, invalid city selection, missing features, invalid accommodation values and successful predictions.

The saved models are experimental and have not been validated against future market conditions.

Predictions should therefore be treated as estimates rather than guaranteed market prices.