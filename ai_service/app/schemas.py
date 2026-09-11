from typing import Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    material: str
    confidence: float
    estimated_price_per_kg: Optional[float] = None
    estimated_weight: Optional[float] = None
    estimated_value: Optional[float] = None