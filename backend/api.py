from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from malnutrition_predictor import MalnutritionPredictor
import joblib
import numpy as np

# ---------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Malnutrition Severity Prediction API")

# Allow CORS (so your React/Angular/Vue frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = MalnutritionPredictor()
predictor.load_model('models/malnutrition_model.pkl')
severity_model = joblib.load('models/malnutrition_severity_model.pkl')
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
#  Root endpoint
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Malnutrition Severity Prediction API is running."}

# ---------------------------------------------------------
#  Image upload & prediction endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    age: str = Form(None),
    height: str = Form(None),
    weight: str = Form(None),
    sex: str = Form(None)
):
    """Accepts an image file, saves it temporarily, runs prediction, and returns results."""
    try:
        # Save uploaded image temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        result, confidence = predictor.predict_new_image(str(temp_path))

        severity_label = None
        # If malnourished, use extra fields to predict severity
        if result == "Malnourished":
            print(f"Received severity fields: age={age}, height={height}, weight={weight}, sex={sex}")
            if None in (age, height, weight, sex):
                os.remove(temp_path)
                raise HTTPException(status_code=422, detail="Missing age, height, weight, or sex for severity prediction.")
            try:
                age_val = int(age)
                height_val = float(height)
                weight_val = float(weight)
                sex_val = int(sex)
            except Exception as conv_err:
                os.remove(temp_path)
                print(f"Conversion error: {conv_err}")
                raise HTTPException(status_code=422, detail=f"Invalid input for severity fields: {conv_err}")
            X_input = np.array([[sex_val, age_val, height_val, weight_val]])
            print(f"Severity model input: {X_input}")
            try:
                severity_label = severity_model.predict(X_input)[0]
            except Exception as model_err:
                os.remove(temp_path)
                print(f"Severity model error: {model_err}")
                raise HTTPException(status_code=500, detail=f"Severity model error: {model_err}")

        # Clean up (optional)
        os.remove(temp_path)

        response = {
            "filename": file.filename,
            "severity_score": confidence,
            "severity_level": result,
            "face_detected": True,
        }
        if severity_label is not None:
            response["malnutrition_type"] = severity_label

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/image")
async def predict_image_photo(file: UploadFile = File(...)):
    """Accepts an image file, saves it temporarily, runs prediction, and returns results."""
    try:
        # Save uploaded image temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run model prediction
        result, confidence = predictor.predict_new_image(str(temp_path))

        # Clean up (optional)
        os.remove(temp_path)

        if result is None:
            raise HTTPException(status_code=400, detail="Could not extract landmarks — ensure face and body are clearly visible in the image.")

        return {
            "filename": file.filename,
            "severity_score": 1.0,
            "severity_level": "Healthy",
            "face_detected": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))