# Rossmann Store Sales Prediction

This repository contains code and resources for predicting sales at Rossmann stores using historical sales data. The project involves applying machine learning models to forecast sales, focusing on factors such as store promotions, competition, holidays, and seasonality.
## Project Overview

Rossmann operates over 3,000 drug stores across 7 European countries. The task is to predict the daily sales of various Rossmann stores across Germany based on historical sales data and other influential factors.
### Key Features:
- **Exploratory Data Analysis (EDA)**: Insights into sales trends, seasonality, and external factors.
- **External Influences**: Integration of holidays, promotions, and competitor data.
- **Time Series Forecasting**: Sales predictions using historical data.
- **Machine Learning Models**: ......
  
<details>
  <summary>Table of Contents</summary>

  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Project Structure](#project-structure)
  - [Data Preprocessing](#data-preprocessing)
  - [Deep Learning Model](#deep-learning-model)
  - [Feature Importance & Confidence Interval](#feature-importance--confidence-interval)
  - [Model Serialization](#model-serialization)
  - [Conclusion](#conclusion)

</details>


## Introduction

Predicting sales for Rossmann stores using Random Forest Regressor and LSTM. Steps include data preprocessing, model building, evaluation, and serialization.

## Dataset

#### The main files are:

- `train.csv`: Historical sales data for Rossmann stores.
- `test.csv`: Data for the test set (stores for which we are predicting sales).
- `store.csv`: Metadata about the stores, such as store type, competition distance, and promotion information.

## Project Structure

- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter Notebooks for EDA, modeling, and evaluation.
- `scripts/`: Python scripts for data preprocessing, feature engineering, and model training.
- `tests/`: Unittest scripts for testing a python module
## Project Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   ├──customer_purchasing_behavior_EDA.ipynb
│   ├──data_preprocessing_pipeline.ipynb
│   └── README.md
├── tests/
│   ├── __init__.py
├── data/
│   ├── store.csv
│   ├── test.csv and train.csv
└── scripts/
    ├── __init__.py
    └── customer_behavior_EDA.py
    |__ Data_preprocessing_pipeline.py
    |__ README.md
```
## Data Preprocessing

1. **Loading Data**: Load store, train, and test datasets.
2. **Datetime Conversion**: Convert `Date` columns to datetime format.
3. **Label Encoding**: Encode categorical features.
4. **Feature Engineering**: Extract year, month, day, week, day of the year, and weekend indicator.
5. **Holiday Proximity**: Calculate days to/from holidays.
6. **Scaling**: Standardize features.

