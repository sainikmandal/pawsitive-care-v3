from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import sys
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Add ml_model directory to Python path ---
# This allows importing DiseasePredictionModel from the ml_model sibling directory
# Assumes main.py is in backend/app/ and train_model.py is in backend/ml_model/
ML_MODEL_DIR = Path(__file__).parent.parent / 'ml_model'
sys.path.append(str(ML_MODEL_DIR.parent)) # Adds 'backend' to path

try:
    from ml_model.train_model import DiseasePredictionModel # Corrected import path
except ImportError as e:
    logger.error(f"Could not import DiseasePredictionModel. Ensure ml_model.train_model exists and backend directory is in sys.path. Error: {e}", exc_info=True)
    # Depending on deployment, you might want the app to fail startup here
    # For now, we'll let it proceed but model loading will fail.
    DiseasePredictionModel = None 


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pawsitive Care API",
    description="API for predicting livestock diseases based on symptoms and other factors.",
    version="1.0.1"
)

# --- Model Loading ---
model_predictor = None
if DiseasePredictionModel:
    try:
        SAVED_MODEL_PATH = ML_MODEL_DIR / "saved_model"
        model_predictor = DiseasePredictionModel()
        model_predictor.load_model_artifacts(SAVED_MODEL_PATH)
        logger.info(f"Disease prediction model loaded successfully from {SAVED_MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model artifacts not found at {SAVED_MODEL_PATH}. API will not be able to make predictions. Please train the model first.")
        model_predictor = None # Ensure model is None if loading failed
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}", exc_info=True)
        model_predictor = None
else:
    logger.error("DiseasePredictionModel class not available due to import error. Prediction endpoint will be disabled.")


# --- Pydantic Models for API Request/Response ---
class PredictionInput(BaseModel):
    animal: str = Field(..., example="cow", description="Type of animal (e.g., cow, sheep).")
    age: float = Field(..., example=3.5, description="Age of the animal in years.")
    temperature: float = Field(..., example=102.5, description="Body temperature of the animal in Fahrenheit (or Celsius, ensure consistency with training).")
    symptoms: List[str] = Field(..., min_length=1, example=["fever", "loss of appetite", "lameness"], description="List of observed symptoms.")
    # Add any other raw features your model expects before preprocessing by the API
    # For example, if you had 'Breed' as a raw input:
    # breed: Optional[str] = Field(None, example="holstein")

class PredictionOutput(BaseModel):
    predicted_disease: str = Field(..., example="foot and mouth", description="The most likely predicted disease.")
    confidence: float = Field(..., example=0.85, description="Confidence score of the prediction (0.0 to 1.0).")
    # You could add probabilities for all classes if needed:
    # class_probabilities: Optional[Dict[str, float]] = None

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Pawsitive Care Disease Prediction API. Visit /docs for API documentation."}

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint to verify API status and model loading."""
    model_status = "loaded" if model_predictor and model_predictor.model else "not loaded"
    return {"status": "healthy", "model_status": model_status}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_disease(input_data: PredictionInput = Body(...)):
    """
    Predicts the most likely disease based on animal type, age, temperature, and symptoms.
    """
    if not model_predictor or not model_predictor.model:
        logger.error("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not available. Please try again later or contact support.")

    try:
        logger.info(f"Received prediction request: Animal={input_data.animal}, Age={input_data.age}, Temp={input_data.temperature}, Symptoms={input_data.symptoms}")

        # --- Map API input to DataFrame structure expected by preprocess_data ---
        # The DiseasePredictionModel.preprocess_data expects 'Symptom 1', 'Symptom 2', 'Symptom 3'
        symptom_dict = {}
        for i in range(3): # Expecting up to 3 symptoms
            if i < len(input_data.symptoms):
                symptom_dict[f'Symptom {i+1}'] = input_data.symptoms[i]
            else:
                symptom_dict[f'Symptom {i+1}'] = 'unknown' # Or None, or handle as per your preprocessing logic for missing symptoms

        raw_df_for_prediction = pd.DataFrame([{
            'Animal': input_data.animal,
            'Age': input_data.age,
            'Temperature': input_data.temperature,
            **symptom_dict # Unpack Symptom 1, Symptom 2, Symptom 3
            # Add other raw features from PredictionInput if they exist
            # 'Breed': input_data.breed if input_data.breed else 'unknown',
        }])
        
        logger.debug(f"DataFrame created for prediction: \n{raw_df_for_prediction}")

        # The .predict method in DiseasePredictionModel now handles its own preprocessing
        predicted_disease, confidence = model_predictor.predict(raw_df_for_prediction)
        
        return PredictionOutput(
            predicted_disease=predicted_disease,
            confidence=confidence
        )
    
    except ValueError as ve: # Catch specific errors from model prediction/preprocessing
        logger.warning(f"Validation or Preprocessing Error during prediction: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Input data error: {str(ve)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# To run this app (from the 'backend' directory):
# Ensure you have an __init__.py in the 'app' and 'ml_model' directories.
# Command: uvicorn app.main:app --reload
