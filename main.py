import pathlib
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 1. Initialize FastAPI
app = FastAPI()

# 2. Add CORS Middleware 
# This is critical to allow your HTML frontend to talk to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows connections from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

# 3. Path Management
# Using pathlib ensures that artifacts are found even if you run the script from a different folder.
BASE_DIR = pathlib.Path(__file__).parent.resolve()

model_path = BASE_DIR / 'loan_model.pkl'
scaler_path = BASE_DIR / 'scaler.pkl'
features_path = BASE_DIR / 'feature_names.pkl'

# 4. Load saved artifacts
# These MUST be generated first by running your updated loan_pred.py.
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    print(f"✓ Backend: Loaded {len(feature_names)} features: {feature_names}")
except Exception as e:
    print(f"✗ Backend Error: Artifacts not found. Run your training script first. {e}")

# 5. Data Model for Input Validation
# This matches the structure expected from your index.html formData.
class LoanInput(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int

# 6. API Endpoints
@app.get("/")
def serve_home():
    """Serves the frontend directly from the backend directory."""
    html_file = BASE_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "Backend is running. Please open index.html manually."}

@app.post("/predict")
async def predict_loan(input_data: LoanInput):
    """Orchestrates the prediction workflow and handles errors."""
    try:
        # Convert input to dictionary
        data_dict = input_data.dict()
        
        # CRITICAL: Reorder inputs to match the EXACT order used during training.
        # If 'ApplicantIncome' was the 6th column in training, it must be the 6th here.
        ordered_input = [data_dict[col] for col in feature_names]
        
        # Scale the data using the saved StandardScaler
        scaled_input = scaler.transform([ordered_input])
        
        # Perform prediction and get probability
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        
        # Return results in the format your index.html expects
        return {
            "status": "Approved" if prediction == 1 else "Rejected",
            "confidence": f"{round(probability * 100, 2)}%"
        }
    except Exception as e:
        # Log the error to the VS Code terminal for debugging
        print(f"PREDICTION ERROR: {str(e)}")
        # Send error to UI so the user doesn't see "undefined"
        return {
            "error": str(e),
            "status": "System Error",
            "confidence": "0%"
        }