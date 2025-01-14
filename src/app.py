from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io
import os
import logging

# Initialize FastAPI and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your trained model
MODEL_PATH = "../model/random_forest_model-12-01-2025-21-38-58.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Ensure the path is correct.")

try:
    rfc_model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the model: {e}")
    raise

# Define the required columns for prediction
required_columns = [
    'Id', 'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2', 'Weekday', 'IsWeekend', 'Day',
    'Month', 'Year', 'IsHoliday', 'StoreType_b', 'StoreType_c',
    'StoreType_d', 'Assortment_b', 'Assortment_c'
]

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "API is up and running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a CSV file, preprocesses it, and returns predictions.
    """
    try:
        # Read the file contents and load it into a DataFrame
        contents = await file.read()
        test_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Check if all required columns exist in the uploaded CSV
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {', '.join(missing_columns)}"})

        # Ensure 'Id' column is present
        if 'Id' not in test_df.columns:
            test_df['Id'] = range(1, len(test_df) + 1)

        # Validate numeric columns
        numeric_columns = [col for col in required_columns if col != 'Id']
        if not all(test_df[col].dtype.kind in 'if' for col in numeric_columns):
            return JSONResponse(status_code=400, content={"error": "Non-numeric values found in required numeric columns."})

        # Make predictions
        predictions = rfc_model.predict(test_df.drop(columns='Id'))

        # Format results
        results = [{"Id": int(id_val), "Prediction": float(pred)} for id_val, pred in zip(test_df['Id'], predictions)]

        # Return the results
        return JSONResponse(content={"Predictions": results})

    except pd.errors.ParserError:
        return JSONResponse(status_code=400, content={"error": "Failed to parse the CSV file."})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": f"Value error: {str(ve)}"})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
