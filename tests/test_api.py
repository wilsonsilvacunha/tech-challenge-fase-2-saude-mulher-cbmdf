import pytest
from pydantic import ValidationError

from api.app import PreverRequest, PreverResponse, app, home, prever
from src.services.predictor import predictor


def test_predictor_service_is_loaded() -> None:
    assert predictor.modelo is not None
    assert predictor.vectorizer is not None


def test_home_returns_status_message() -> None:
    assert home() == {"mensagem": "API de diagnóstico funcionando 🚀"}


def test_prever_returns_diagnostico_for_valid_payload() -> None:
    response = prever(PreverRequest(texto="corrimento com odor forte"))

    assert isinstance(response, PreverResponse)
    assert isinstance(response.diagnostico, str)
    assert response.diagnostico


def test_prever_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        PreverRequest(texto="")


def test_openapi_exposes_prever_request_response_and_validation() -> None:
    spec = app.openapi()

    assert "/prever" in spec["paths"]
    post_spec = spec["paths"]["/prever"]["post"]
    assert post_spec["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PreverRequest"
    }
    assert post_spec["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PreverResponse"
    }
    assert "422" in post_spec["responses"]
