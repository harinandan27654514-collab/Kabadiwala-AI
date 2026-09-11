from pathlib import Path
from ultralytics import YOLO


# Locate the trained model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "best.pt"
)


# Load the trained YOLO model
model = YOLO(str(MODEL_PATH))


# Sample scrap prices per kilogram
PRICE_PER_KG = {
    "Paper": 15,
    "Plastic": 30,
    "Glass": 10,
    "Metal": 40,
    "Organic": 5,
    "Electronics": 200,
    "Miscellaneous": 10,
}


def predict_material(image_path: str):
    results = model(
        image_path,
        conf=0.25,
        verbose=False
    )

    # If no result is returned
    if not results:
        return {
            "material": "Unknown",
            "confidence": 0.0,
            "estimated_price_per_kg": None,
            "estimated_weight": None,
            "estimated_value": None,
        }

    result = results[0]

    # If no object is detected
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "material": "Unknown",
            "confidence": 0.0,
            "estimated_price_per_kg": None,
            "estimated_weight": None,
            "estimated_value": None,
        }

    # Find the detection with the highest confidence
    best_box_index = int(
        result.boxes.conf.argmax().item()
    )

    # Get class ID and confidence
    class_id = int(
        result.boxes.cls[best_box_index].item()
    )

    confidence = float(
        result.boxes.conf[best_box_index].item()
    )

    # Convert class ID into material name
    material = model.names[class_id]

    # Get sample price
    price = PRICE_PER_KG.get(material)

    return {
        "material": material,
        "confidence": round(confidence, 4),
        "estimated_price_per_kg": price,
        "estimated_weight": None,
        "estimated_value": None,
    }