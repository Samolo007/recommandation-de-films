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


class Movie(BaseModel):
    item_id: int
    title: str

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

NOT_FOUND = "Film introuvable : cet identifiant n'existe pas dans le catalogue."
NO_NEIGHBORS = "Pas de recommandation : aucun spectateur n'a donné 4 ou 5 à ce film"


@app.post("/predict", response_model=RecommendationResponse)
def predict(movie: MovieRequest, request: Request):
    index = request.app.state.index
    catalog = index["catalog"]
    neighbors = index["neighbors"]

    if movie.item_id not in catalog:
        raise HTTPException(status_code=404, detail=NOT_FOUND)

    title = catalog[movie.item_id]
    if movie.item_id not in neighbors:
        return RecommendationResponse(
            item_id=movie.item_id, title=title, recommendations=[], message=NO_NEIGHBORS
        )

    return RecommendationResponse(
        item_id=movie.item_id, title=title, recommendations=neighbors[movie.item_id]
    )

@app.get("/movies", response_model=list[Movie])
def movies(request: Request):
    catalog = request.app.state.index["catalog"]
    return [
        Movie(item_id=item_id, title=title)
        for item_id, title in sorted(catalog.items(), key=lambda pair: pair[1])
    ]