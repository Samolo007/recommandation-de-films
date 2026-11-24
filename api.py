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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL.is_file():
        raise RuntimeError(
            "Index introuvable. Lancez d'abord : python src/train_model.py"
        )
    app.state.index = joblib.load(MODEL)
    yield


app = FastAPI(title="Recommandation de films", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    details = [
        {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health():
    return {"status": "ok"}