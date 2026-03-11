# Uber Fare Analysis & Prediction



![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)

![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)

![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)



> End-to-end data analysis and machine learning project on Uber ride data — covering data cleaning, feature engineering, exploratory data analysis, and fare prediction using Random Forest Regression.



---



##  Table of Contents



- [Overview](#overview)

- [Dataset](#dataset)

- [Project Structure](#project-structure)

- [Installation](#installation)

- [Pipeline Walkthrough](#pipeline-walkthrough)

- [Exploratory Data Analysis](#exploratory-data-analysis)

- [Machine Learning Model](#machine-learning-model)

- [Key Findings](#key-findings)

- [Results](#results)

- [Technologies Used](#technologies-used)

- [Future Improvements](#future-improvements)



---



## Overview



This project performs a comprehensive analysis of Uber ride data to uncover patterns in revenue, ride frequency, and trip behavior and builds a predictive model to estimate fare amounts based on trip features.



The analysis pipeline includes:

- **Data Ingestion & Cleaning** — handling nulls, duplicates, invalid coordinates, and negative fares

- **Feature Engineering** — extracting temporal features and computing Euclidean trip distance

- **EDA & Visualization** — revenue trends by day, hour, year, and passenger count

- **Regression Modeling** — Random Forest with preprocessing pipeline achieving ~74.6% R² score






## Project Structure



```

uber-fare-analysis/

│

├── uber_data.csv              # Raw dataset

├── Final.csv                  # Cleaned & feature-engineered dataset

├── analysis.ipynb             # Main Jupyter Notebook

├── README.md                  # Project documentation

│

└── outputs/

    ├── day_of_week_revenue.png

    ├── year_wise_revenue.png

    ├── hourly_revenue.png

    ├── passenger_revenue.png

    └── distance_vs_fare.png

```



---



## Installation



**1. Clone the repository**

```bash

git clone https://github.com/yourusername/uber-fare-analysis.git

cd uber-fare-analysis

```



**2. Create a virtual environment (recommended)**

```bash

python -m venv venv

source venv/bin/activate        # On Windows: venv\Scripts\activate

```



**3. Install dependencies**

```bash

pip install -r requirements.txt

```



**4. Launch Jupyter Notebook**

```bash

jupyter notebook analysis.ipynb

```



### Requirements



```

pandas

numpy

matplotlib

seaborn

scikit-learn

jupyter

```



---



## Pipeline Walkthrough



### 1. Data Cleaning



- Renamed ambiguous columns (`key` → `Ride_Time_Stamp`, later dropped)

- Removed null rows (1 detected), duplicates, and zero/negative fares (22 rows)

- Filtered out GPS coordinates outside valid lat/lon ranges (12 rows)

- Removed a record with an impossible passenger count of 208

- Cast `pickup_datetime` to datetime, stripped timezone, and `fare_amount` to numeric



### 2. Feature Engineering



New columns derived from `pickup_datetime`:



| Feature | Description |

|---------|-------------|

| `Day_of_Week` | Monday–Sunday |

| `Year` | Calendar year |

| `Month` | Month number |

| `Day` | Day of month |

| `Hour` | Hour of day (0–23) |

| `Distance` | Euclidean distance (latitude/longitude delta) |



### 3. Anomaly Flags



| Issue | Count |

|-------|-------|

| Rides with 0 passengers | 708 |

| Rides with zero/negative distance | 5,632 |

| Rides with invalid fare | 5 |



---



## Exploratory Data Analysis



### Day of Week vs. Revenue

Revenue trends across weekdays — useful for identifying peak business days.







### Year-Wise Revenue Distribution

Pie chart showing proportional revenue contribution per year (2009–2015).


### Hourly Revenue Pattern

Identifies peak hours for ride demand and revenue.



### Passenger Count vs. Revenue

Solo riders (1 passenger) dominate both ride count and total revenue.



| Passengers | Total Revenue ($) |

|------------|------------------|

| 1 | 1,557,794 |

| 2 | 346,783 |

| 5 | 157,046 |

| 3 | 102,093 |

| 6 | 51,929 |

| 4 | 49,769 |

| 0 | 6,683 |



### Distance vs. Fare (Cleaned)

Scatter plot filtered to distances 0–50 miles and fares $0–$200 to remove outliers.


**Trip Distance Distribution:**



| Trip Type | Share |

|-----------|-------|

| Short (< 2 mi) | 99.78% |

| Long (> 10 mi) | 0.22% |

| Medium (2–10 mi) | 0.004% |



---



## Machine Learning Model



### Linear Regression (Baseline)



Trained on the cleaned, filtered dataset (Distance 0–50 mi, Fare $0–200):



```

Intercept  (Base Fare)  : $11.26

Coefficient (per mile)  : $2.18

```



**Example Predictions:**

- 5 miles  → **$22.16**

- 10 miles → **$33.06**

- 20 miles → **$55.86**



This closely mirrors real NYC taxi pricing (~$2–3 per mile).



---



### Random Forest Regressor (Final Model)



A full ML pipeline with preprocessing:



```python

Features Used:

  - Distance

  - Hour

  - is_weekend (engineered)

  - passenger_count

  - pickup_zone (KMeans clustering, k=12)



Preprocessing:

  - StandardScaler on numerical features

  - OneHotEncoder on categorical features



Model:

  RandomForestRegressor(n_estimators=200, max_depth=12)

```



**Train/Test Split:** 80% / 20%



| Metric | Score |

|--------|-------|

| Mean Absolute Error (MAE) | **$2.32** |

| R² Score | **0.746** |



---



## Key Findings



- **Solo rides dominate** — single-passenger trips account for ~69% of total revenue

- **Base fare is ~$11.26** — consistent with NYC minimum fare + surcharges

- **~99.8% of all trips are under 2 miles** — typical short urban hops

- **Average trip distance** is only 0.20 miles (Euclidean), suggesting dense pickup/dropoff clustering

- **Zero-passenger rides (708 records)** likely represent test rides or data entry errors

- **Revenue is spread across hours** with a visible peak during morning/evening commute windows



---



## Results



| Model | MAE | R² |

|-------|-----|----|

| Linear Regression | ~$3.50 (est.) | ~0.55 (est.) |

| Random Forest (Final) | **$2.32** | **0.746** |



The Random Forest model explains ~74.6% of fare variance, with an average prediction error of $2.32 — competitive for a dataset heavily influenced by geographic and temporal noise.



---



## Technologies Used



| Tool | Purpose |

|------|---------|

| `pandas` | Data manipulation and aggregation |

| `numpy` | Numerical computing and distance calculation |

| `matplotlib` | Base plotting |

| `seaborn` | Statistical visualizations |

| `scikit-learn` | ML pipeline, preprocessing, model training |

| `KMeans` | Geospatial clustering of pickup zones |

| `jupyter` | Interactive development environment |



---



## Future Improvements



- [ ] Use Haversine formula for accurate geodesic distance instead of Euclidean

- [ ] Incorporate external data (weather, NYC events, traffic) for richer features

- [ ] Experiment with XGBoost or LightGBM for improved accuracy

- [ ] Add SHAP values for model explainability

- [ ] Build an interactive dashboard using Streamlit or Plotly Dash

- [ ] Deploy the model via a REST API (FastAPI + Docker)




## Author

**Manjunath Kaaluru**



> ⭐ If you found this project useful, consider giving it a star on GitHub!
