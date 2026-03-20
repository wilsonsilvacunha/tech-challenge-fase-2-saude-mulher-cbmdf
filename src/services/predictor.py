from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


class PredictorService:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.modelo = joblib.load(models_dir / "modelo.pkl")
        self.vectorizer = joblib.load(models_dir / "vectorizer.pkl")

    def predict(self, texto: str) -> str:
        vetor = self.vectorizer.transform([texto])
        resultado = self.modelo.predict(vetor)
        return str(resultado[0])


predictor = PredictorService(MODELS_DIR)
