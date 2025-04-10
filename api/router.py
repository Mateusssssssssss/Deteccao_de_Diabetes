# APIRouter: separar e agrupar rotas.
from fastapi import APIRouter, HTTPException
from api.models import InputData
from api.predict import prediction

router = APIRouter()

@router.post("/predict")
def predict(data: InputData):
    try:
        return prediction(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erro interno na previsão")
