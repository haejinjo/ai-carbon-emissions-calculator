from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from ai_emissions_calculator import AIEmissionsCalculator  # your full class in emissions.py

app = FastAPI()
calculator = AIEmissionsCalculator()

class EmissionsRequest(BaseModel):
    model_size: str
    tokens_processed: int
    instance_type: str
    region: str
    provider: str
    task_type: Optional[str] = 'inference'
    batch_size: Optional[int] = 1
    duration_hours: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    watttime_username: Optional[str] = None
    watttime_password: Optional[str] = None

@app.post("/estimate_emissions")
def estimate_emissions(payload: EmissionsRequest):
    result = calculator.estimate_emissions(**payload.dict())
    return result
