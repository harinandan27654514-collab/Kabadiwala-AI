import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import predict_material
from .schemas import PredictionResponse


app = FastAPI(
    title="Kabadiwala AI Service",
    description="AI service for scrap material recognition",
    version="1.0.0"
)


# Allow the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Kabadiwala AI Service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    suffix = os.path.splitext(
        file.filename or ".jpg"
    )[1]

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temporary_path = temporary_file.name

    try:
        # Save uploaded image temporarily
        with temporary_file as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run AI prediction
        prediction = predict_material(temporary_path)

        return prediction

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:
        # Delete temporary image
        if os.path.exists(temporary_path):
            os.remove(temporary_path)