from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

# Load your trained model
rfc_model = joblib.load("/home/gech/10 acadamy/week-4/Rossmann--Sales-Prediction/model/random_forest_model-12-01-2025-21-38-58.pkl")

app = FastAPI()

# Define the required columns for prediction
required_columns = [
    'Id', 'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2', 'Weekday', 'IsWeekend', 'Day',
    'Month', 'Year', 'IsHoliday', 'StoreType_b', 'StoreType_c',
    'StoreType_d', 'Assortment_b', 'Assortment_c'
]
