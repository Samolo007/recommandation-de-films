from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "knn_index.joblib"


class MovieRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_id: int = Field(gt=0)              


class Recommendation(BaseModel):
    item_id: int                            
    title: str                              
    score: float                            


class RecommendationResponse(BaseModel):
    item_id: int
    title: str
    recommendations: list[Recommendation]    
    message: str | None = None              