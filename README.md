# Wildfire Risk Prediction Model for Nepal

## Overview

This project develops a machine-learning system that estimates **next-day wildfire hotspot risk across Nepal**.

The model uses historical environmental conditions from ERA5-Land together with NASA FIRMS active-fire detections. It learns the relationship between current and recent weather conditions and whether a fire hotspot is detected in the same geographic grid cell on the following day.

The final system contains two models:

1. **Full environmental benchmark model**
2. **Hardware-compatible deployment model**

The hardware-compatible model is intended to run on the Raspberry Pi connected to the ESP8266 wildfire-monitoring sensor network.

---

## What the Model Predicts

For every geographic grid cell, the model answers:

> Based on today's environmental conditions and recent environmental history, what is the risk that NASA FIRMS will detect an active-fire hotspot in this same area tomorrow?

The prediction target is:

```text
fire_next_day
```

The target equals:

- `1` when a FIRMS hotspot is detected in the same grid cell on the following day
- `0` when no hotspot is detected in that grid cell on the following day

The model produces a **risk score** and an operational risk level:

```text
Low
Moderate
High
Very High
```

The risk score is a machine-learning score used for ranking and alerting. It should not be interpreted as an exact physical probability of wildfire ignition.

---

## Dataset

The corrected modelling dataset is:

```text
wildfire_base_grid_daily_same_day_full_rerun_v2.parquet
```

It contains one row for every:

```text
date × ERA5-Land grid cell
```

The corrected dataset includes:

- 500 dates
- 1,361 ERA5-Land grid cells
- 680,500 grid-cell daily observations
- data from 5 January 2025 through 19 May 2026
- corrected FIRMS fire labels through May 2026

The notebook validates that March-May 2026 contains positive fire labels before model training begins.

---

## Data Sources

### ERA5-Land

ERA5-Land provides daily environmental conditions for each geographic grid cell, including:

- mean temperature
- maximum temperature
- dew point
- relative humidity
- wind components
- wind speed
- precipitation
- soil moisture

### NASA FIRMS

NASA FIRMS provides active-fire hotspot detections.

Each FIRMS detection is spatially matched to the nearest ERA5-Land grid cell. Multiple detections in the same grid cell on the same date are aggregated into one daily fire label.

---

## Next-Day Target Construction

The original dataset contains a same-day field:

```text
fire_detected
```

The model-training notebook shifts this field by one day within each grid cell:

```python
wildfire["fire_next_day"] = (
    wildfire
    .groupby(["latitude", "longitude"])["fire_detected"]
    .shift(-1)
)
```

This creates the modelling relationship:

```text
Environmental conditions on day t
                ↓
FIRMS hotspot detection on day t + 1
```

The last date from each grid cell is removed because it has no following-day label.

---

## Feature Engineering

Wildfire risk depends on both current weather and accumulated environmental conditions. The notebook therefore creates rolling historical features separately for every grid cell.

### Current-day features

The full environmental model uses:

- latitude
- longitude
- mean temperature
- maximum temperature
- dew point
- relative humidity
- east-west wind
- north-south wind
- wind speed
- precipitation
- soil moisture

### Rolling environmental features

The model creates:

- 3-day rainfall sum
- 7-day rainfall sum
- 14-day rainfall sum
- 3-day mean humidity
- 7-day mean humidity
- 3-day mean temperature
- 7-day mean temperature
- 3-day maximum temperature
- 7-day maximum temperature
- 3-day mean dew point
- 7-day mean dew point
- 3-day maximum wind speed
- 7-day mean wind speed
- 7-day mean soil moisture
- consecutive dry days

### Seasonal features

The model also includes:

- month
- day-of-year sine
- day-of-year cosine
- winter indicator
- pre-monsoon indicator
- monsoon indicator
- post-monsoon indicator

Cyclical day-of-year features allow the model to represent annual seasonality continuously.

---

## Target-Leakage Prevention

FIRMS-derived variables describe an already observed fire event and are therefore excluded from model inputs.

The following fields are retained only for analysis:

```text
fire_detected
fire_next_day
fire_count
frp_sum
frp_max
confidence_mean
brightness_mean
match_distance_mean_km
match_distance_max_km
```

This prevents the model from receiving information that would reveal the prediction target.

---

## Models

## 1. Full Environmental Benchmark Model

The full model uses the complete ERA5-Land environmental feature set.

Its purpose is to:

- provide the strongest regional benchmark
- produce Nepal-wide next-day wildfire-risk maps
- measure the predictive value of the full environmental dataset
- support academic evaluation and comparison

The saved full model is:

```text
wildfire_full_environmental_next_day_v1.joblib
```

---

## 2. Hardware-Compatible Model

The hardware-compatible model uses only features that can be reproduced from the deployed sensor system.

Its inputs are based on:

- DHT11 temperature
- DHT11 humidity
- calculated dew point
- GPS or configured deployment coordinates
- daily sensor summaries
- 3-day sensor history
- 7-day sensor history
- date and seasonal information

The hardware model uses:

```text
latitude
longitude
temperature_mean_c
temperature_max_c
dewpoint_c
relative_humidity
humidity_mean_3d
humidity_mean_7d
temperature_mean_3d
temperature_mean_7d
temperature_max_3d
temperature_max_7d
dewpoint_mean_3d
dewpoint_mean_7d
month
day_of_year_sin
day_of_year_cos
season_winter
season_pre_monsoon
season_monsoon
season_post_monsoon
```

The final deployment model is:

```text
wildfire_hardware_next_day_v1.joblib
```

---

## Why Flame and MQ2 Are Not ML Features

The ESP8266 hardware sends:

- temperature
- humidity
- MQ2 reading
- flame status
- latitude
- longitude
- GPS-fix status

However, the historical ERA5-FIRMS dataset does not contain matching historical MQ2 or flame-sensor measurements.

For that reason:

```text
MQ2 is not used as an ML feature.
Flame status is not used as an ML feature.
```

They remain part of the immediate reactive safety layer.

The final system separates:

```text
Predictive layer:
Next-day wildfire risk model

Reactive layer:
Flame alarm and MQ2 smoke warning
```

A low ML risk score must never suppress a flame or smoke alert.

---

## Chronological Data Splitting

The data is divided chronologically instead of randomly.

```text
Training:
5 January 2025 to 31 December 2025

Validation:
1 January 2026 to 28 February 2026

Testing:
1 March 2026 to 18 May 2026
```

This prevents future observations from being used to predict earlier observations.

The notebook refuses to continue unless training, validation, and testing all contain positive next-day fire examples.

---

## Class Imbalance

Wildfire hotspot observations are rare compared with non-fire grid-cell days.

The Random Forest uses:

```text
class_weight = balanced_subsample
```

This increases the importance of the minority fire class during model training.

The final notebook uses the complete chronological training dataset by default. A reduced negative-class sample may be used only for temporary diagnostic runs on limited Colab hardware.

---

## Model Algorithm

Both models use a Random Forest classifier.

Random Forest is suitable because it can learn:

- non-linear relationships
- interactions between weather variables
- geographic patterns
- seasonal patterns
- accumulated dry and hot conditions

Each model produces a continuous wildfire risk score using:

```python
model.predict_proba(features)[:, 1]
```

---

## Risk Thresholds

Risk thresholds are selected using only the validation dataset.

The notebook derives:

```text
Moderate:
High-recall validation threshold

High:
Stricter validation threshold

Very High:
Validation F1-optimised threshold
```

The untouched test dataset is used only for final evaluation.

The thresholds are stored inside the model bundle so that the Raspberry Pi uses exactly the same operating thresholds as the training notebook.

---

## Evaluation

The notebook evaluates both models using:

- precision
- recall
- F1-score
- average precision
- ROC-AUC
- confusion matrix
- precision-recall curve
- feature importance

Accuracy is not used as the main metric because the dataset is highly imbalanced.

The notebook compares:

```text
Full environmental model
versus
Hardware-compatible model
```

The final deployment decision should be based on the hardware model's chronological test performance.

---

## Spatial Risk Prediction

The trained models can be applied to all 1,361 ERA5-Land grid cells for the latest available date.

For every grid cell, the output contains:

```text
date
latitude
longitude
risk score
risk level
```

The notebook saves a ranked next-day risk surface across Nepal.

---

## Saved Outputs

After successful execution, the notebook creates the following files.

### Model files

```text
models/
├── wildfire_full_environmental_next_day_v1.joblib
└── wildfire_hardware_next_day_v1.joblib
```

### Evaluation and deployment files

```text
results/
├── model_comparison_v2.csv
├── risk_level_metrics_v1.csv
├── model_metrics_v1.json
├── hardware_model_contract_v1.json
├── era5_training_grid_v1.csv
├── hardware_feature_example_v1.json
├── full_model_feature_importance_v2.csv
├── hardware_model_feature_importance_v2.csv
├── hardware_model_test_confusion_matrix_v2.png
├── model_comparison_test_precision_recall_v2.png
├── hardware_model_feature_importance_top15_v2.png
├── latest_hardware_next_day_risk_v1.csv
└── latest_hardware_next_day_risk_v1.parquet
```

---

## Hardware Model Bundle

The hardware model bundle stores:

- trained Random Forest classifier
- exact feature-column order
- risk thresholds
- prediction target
- forecast horizon
- training and evaluation periods
- training-grid coordinates
- required sensor-history length
- sensor data-quality rules
- feature definitions
- scikit-learn version
- model version

Example loading code:

```python
import joblib

bundle = joblib.load(
    "wildfire_hardware_next_day_v1.joblib"
)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
risk_thresholds = bundle["risk_thresholds"]
training_grid = bundle["training_grid"]
```

---

## Required Hardware Data Processing

Before the model can run on the Raspberry Pi, the three-second sensor readings must be converted into daily model features.

For each sender, the Raspberry Pi must calculate:

```text
Daily mean temperature
Daily maximum temperature
Daily mean humidity
Daily mean dew point

3-day mean temperature
7-day mean temperature

3-day maximum temperature
7-day maximum temperature

3-day mean humidity
7-day mean humidity

3-day mean dew point
7-day mean dew point
```

The system requires seven days of valid history before producing a complete prediction.

Invalid DHT11 readings must be removed when:

```text
temperature == -1
humidity == -1
temperature or humidity is missing
humidity is outside 0-100%
```

The deployment contract currently requires at least 80% valid readings in the daily window.

---

## Location Handling

The hardware model is trained on ERA5-Land grid coordinates, not arbitrary GPS coordinates.

The Raspberry Pi must select location in this order:

```text
1. Current valid GPS position
2. Last valid GPS position
3. Sender-specific configured deployment coordinate
```

The selected sensor coordinate must then be matched to the nearest coordinate stored in:

```text
era5_training_grid_v1.csv
```

The matched grid latitude and longitude are used as model features.

Each sender should have its own fallback deployment coordinate.

---

## Intended Hardware Workflow

```text
ESP8266 sender
    ↓ every 3 seconds
Temperature, humidity, MQ2, flame, GPS
    ↓
ESP8266 receiver
    ↓
Raspberry Pi
    ├── Immediate flame and MQ2 processing
    ├── Firebase sensor updates
    ├── Sensor-history storage
    └── Daily feature aggregation
            ↓
wildfire_hardware_next_day_v1.joblib
            ↓
Next-day risk score and risk level
            ↓
Firebase ML prediction
            ↓
Dashboard and predictive-risk notification
```

---

## Final Alert Priority

The final application should use this order:

```text
1. Flame detected
   → Immediate fire alarm

2. MQ2 above threshold
   → Immediate smoke warning

3. Very-high ML risk
   → Elevated next-day wildfire-risk warning

4. High ML risk
   → Increased next-day wildfire-risk status

5. Otherwise
   → Normal predictive-risk status
```

The predictive model complements the hardware alarm system. It does not replace it.

---

## Running the Training Notebook

Open:

```text
Wildfire_Model_Training_Final.ipynb
```

Run all cells from top to bottom.

Before accepting the final model, confirm that:

- the corrected Parquet loads
- all three chronological periods contain positive labels
- both models train successfully
- test metrics are finite
- the hardware model reloads successfully
- saved and in-memory predictions match
- the hardware deployment files are created

---

## Current Project Status

Completed:

- ERA5-Land and FIRMS data preparation
- corrected March-May FIRMS labels
- next-day target construction
- feature engineering
- leakage prevention
- chronological model design
- full benchmark model
- hardware-compatible model design
- deployment model bundle
- risk-threshold design
- output and contract definitions

Next stage:

```text
Raspberry Pi model implementation
```

The implementation stage will:

1. store and validate incoming sensor readings
2. aggregate daily temperature and humidity features
3. calculate dew point
4. maintain 3-day and 7-day sensor history
5. match each sensor to the nearest ERA5 grid cell
6. load the saved hardware model
7. generate the next-day wildfire risk
8. upload predictions to Firebase
9. preserve immediate flame and MQ2 alerts
