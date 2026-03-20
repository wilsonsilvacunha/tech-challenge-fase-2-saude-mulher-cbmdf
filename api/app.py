from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.services.predictor import predictor

app = FastAPI()


class PreverRequest(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto com os sintomas relatados.")


class PreverResponse(BaseModel):
    diagnostico: str


@app.get("/")
def home():
    return {"mensagem": "API de diagnóstico funcionando 🚀"}

@app.post("/prever", response_model=PreverResponse)
def prever(dados: PreverRequest) -> PreverResponse:
    diagnostico = predictor.predict(dados.texto)
    return PreverResponse(diagnostico=diagnostico)
