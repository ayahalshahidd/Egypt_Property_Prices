# Egypt Property Prices: Spark-Based Predictive Pricing Pipeline

## Requirement Compliance Summary

| Requirement from project brief | Project status | Evidence |
| --- | --- | --- |
| Innovative idea with business value | Satisfied | Predicts Egyptian real-estate price per square meter to support buyer, investor, and agency pricing decisions. |
| Dataset large enough for big-data processing | Satisfied | Raw `propertyfinder.csv` is about 69 MB, with 39,713 initial rows. |
| Use a big-data framework | Satisfied | Apache Spark/PySpark is used for ingestion, preprocessing, feature engineering, model training, tuning, evaluation, and prediction. |
| Pseudo-distributed mode | Satisfied | Spark defaults to `local-cluster[2,2,2048]`, a single-machine multi-worker Spark mode, with fallback to `local[*]`. |
| Final document: problem, pipeline, preprocessing, visualization, insights, training, evaluation, unsuccessful trials, future work | Satisfied in this document | Sections below follow the requested final-document structure. |
| Train, validation, and test evaluation | Implemented | The Spark training code now creates explicit train, validation, and test splits and writes metrics for all three after the next run. The latest completed run before this update contains holdout test metrics only. |

## I. Brief Problem Description

Real-estate prices in Egypt vary strongly by location, property type, size, completion status, furnishing, listing quality, and amenities. Buyers, investors, and real-estate agencies need a data-driven way to estimate whether a property is fairly priced instead of relying only on manual comparison.

This project solves a regression problem: predicting `price_per_sqm` for buy listings. The business value is an objective property-pricing assistant that can support valuation, deal screening, market comparison, and investment decision-making.

## II. Project Pipeline

The final pipeline is implemented as a Spark-first project:

1. Load raw CSV data with Spark.
2. Clean and validate listings with Spark DataFrame transformations.
3. Engineer numerical, categorical, location, amenity, and listing-quality features with PySpark.
4. Build Spark ML feature vectors using `VectorAssembler`.
5. Train Spark ML regression models in pseudo-distributed mode.
6. Tune hyperparameters with Spark ML `TrainValidationSplit`.
7. Evaluate models on train, validation, and test data.
8. Save the best Spark model artifact and metadata.
9. Generate reports, metrics files, and visualizations.
10. Generate predictions through Spark model inference.

Main implementation files:

- `scripts/train_model.py`
- `scripts/generate_predictions.py`
- `src/data/load_data.py`
- `src/data/preprocess.py`
- `src/features/build_features.py`
- `src/models/train_spark.py`
- `src/models/predict_spark.py`
- `src/visualization/visualize.py`

## III. Analysis and Solution of the Problem

### 1. Data Preprocessing

The raw dataset is loaded from `data/raw/propertyfinder.csv`. The latest generated validation report shows:

| Metric | Value |
| --- | ---: |
| Initial rows | 39,713 |
| Final cleaned rows | 19,316 |
| Dropped columns | 24 |
| Duplicates removed | 0 |
| Critical null rows removed | 6 |
| Payment-method rows removed | 20,261 |
| Outliers removed | 130 |
| Bedroom median | 3 |
| Bathroom median | 3 |

Cleaning steps:

- Removed personally identifying/contact fields and columns not useful for modeling.
- Removed duplicated listings by `listing_id`.
- Parsed numeric columns such as bedrooms, bathrooms, price, area, and image count.
- Converted special bedroom/bathroom values such as `studio` and `7+`.
- Imputed bedroom and bathroom missing values using approximate Spark medians.
- Filled missing furnishing, completion status, payment method, and image count values.
- Standardized payment methods into `cash`, `installments`, and `both`.
- Created the target variable `price_per_sqm = price_egp / area_value`.
- Removed area and price-per-square-meter outliers using Spark approximate quantiles and area bounds.

### 2. Visualization

The Spark training pipeline generates the following figures under `reports/figures/`:

- `target_distribution.png`: distribution of property price per square meter.
- `spark_model_comparison.png`: model comparison by R2 and RMSE.
- `spark_feature_importance.png`: top predictive features for the selected Spark model.
- `spark_residuals.png`: residuals versus predicted price per square meter.

These visualizations support both technical evaluation and business interpretation.

#### Target Distribution

![Target distribution](figures/target_distribution.png)

#### Spark Model Comparison

![Spark model comparison](figures/spark_model_comparison.png)

#### Spark Feature Importance

![Spark feature importance](figures/spark_feature_importance.png)

#### Spark Residuals

![Spark residuals](figures/spark_residuals.png)

### 3. Extracting Insights from Data

The engineered feature set captures several interpretable business signals:

- Location is represented using latitude, longitude, and target encodings for city, town, district, and property type.
- Property size and structure are represented through area, log area, bedrooms, bathrooms, total rooms, area per room, and area-bedroom interaction.
- Listing quality is represented through premium, verified, direct-from-developer, new-construction, and image-count indicators.
- Amenities are represented through pool, gym, security, parking, garden, balcony, private pool, spa, and air-conditioning flags.
- Market/transaction structure is represented through furnishing, completion status, installments, and days listed.

The strongest business expectation is that location, property type, area, room structure, and premium amenities should explain a significant share of price-per-square-meter variation. The generated Spark feature-importance chart should be used in the presentation to explain which signals mattered most in the selected model.

### 4. Model / Classifier Training

This is a regression problem, not a classification problem. The target is modeled as `log1p(price_per_sqm)` to reduce the impact of highly expensive listings.

The final implementation uses Spark ML only:

| Model | Purpose |
| --- | --- |
| Spark Linear Regression | Distributed linear baseline. |
| Spark Decision Tree Regressor | Non-linear interpretable baseline. |
| Spark Random Forest Regressor | Distributed ensemble model for stronger non-linear performance. |
| Spark Gradient-Boosted Trees | Available in config but disabled by default because it is slow in local pseudo-distributed mode. |

The project uses Spark `TrainValidationSplit` for hyperparameter tuning. Current default tuning includes:

- Linear Regression: `regParam`, `elasticNetParam`
- Decision Tree: `maxDepth`, `minInstancesPerNode`
- Random Forest: `numTrees`, `maxDepth`, `minInstancesPerNode`, `featureSubsetStrategy`
- GBT: configured but disabled by default

## IV. Results and Evaluation

The latest completed Spark run selected Spark Random Forest as the best model by holdout R2.

| Model | Test R2 | Test RMSE log | Test MAE log | Test RMSE EGP/sqm |
| --- | ---: | ---: | ---: | ---: |
| Spark Random Forest | 0.5432 | 0.4393 | 0.3246 | 59,584 |
| Spark Decision Tree | 0.4707 | 0.4729 | 0.3506 | 60,799 |
| Spark Linear Regression | 0.4540 | 0.4803 | 0.3621 | 64,537 |

Because the target is continuous, the project reports regression metrics instead of classification accuracy:

- R2: explained variance, higher is better.
- RMSE on log target: root mean squared error after log transform.
- MAE on log target: mean absolute error after log transform.
- RMSE in EGP/sqm: error transformed back to the original business unit.

The code now creates explicit train, validation, and test splits. After rerunning `python scripts/train_model.py`, the generated `reports/results/spark_model_comparison.csv` and `reports/results/spark_model_experimentation.md` will include train, validation, and test metrics. This directly addresses the final-document requirement for train, validation, and test evaluation.

## V. Unsuccessful Trials Not Included in the Final Solution

Several approaches were intentionally removed or excluded from the final solution:

- Local scikit-learn models were removed because the final requirement emphasizes a big-data-style pipeline.
- Local XGBoost was removed from the production path because it was not Spark-distributed in this implementation.
- Spark Gradient-Boosted Trees was disabled by default because it ran slowly in local pseudo-distributed mode on Windows. It remains configurable in `configs/spark_model_params.json` for optional longer experiments.
- Native Spark model saving initially failed on Windows when `winutils.exe` was missing. This was resolved operationally by configuring `HADOOP_HOME` and was also handled defensively in code so reports and metrics are still written if model persistence fails.
- Notebook-based experiments were removed to keep the deliverable focused on reusable Spark code.

## VI. Enhancements and Future Work

Recommended next improvements:

- Add a dataset source link and license note to the README or appendix.
- Run the updated Spark pipeline once more to refresh train, validation, and test metrics in generated result files.
- Add richer exploratory Spark aggregations by city, district, property type, furnishing, and amenities.
- Enable Spark GBT for a long experiment after reducing its search space further or running on a stronger machine.
- Add model monitoring checks for residual bias by city, district, and price segment.
- Add a small Streamlit or dashboard layer for business users to enter listing attributes and receive predicted price-per-square-meter ranges.
- Move from pseudo-distributed local Spark to a fully distributed Spark cluster for bonus credit.

## Deliverable Checklist

| Deliverable | Status |
| --- | --- |
| Final document | This file. |
| Codes | Present under `src/`, `scripts/`, `configs/`, and `tests/`. |
| Presentation business part | Use problem description, business value, insights, and recommendations from this report. |
| Presentation technical part | Use pipeline, Spark architecture, preprocessing, feature engineering, model training, and evaluation sections from this report. |
