import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "data" / "training_samples.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


def load_dataset(dataset_path: Path) -> tuple[list[str], list[str]]:
    textos: list[str] = []
    diagnosticos: list[str] = []

    with dataset_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            textos.append(row["texto"].strip())
            diagnosticos.append(row["diagnostico"].strip())

    if not textos or not diagnosticos:
        raise ValueError("O dataset de treinamento está vazio.")

    return textos, diagnosticos


def train_and_save(dataset_path: Path = DATASET_PATH) -> None:
    textos, diagnosticos = load_dataset(dataset_path)

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(textos)

    modelo = MultinomialNB()
    modelo.fit(features, diagnosticos)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Dataset: {dataset_path}")
    print(f"Amostras: {len(textos)}")
    print(f"Classes: {len(modelo.classes_)}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Vectorizer salvo em: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_and_save()
