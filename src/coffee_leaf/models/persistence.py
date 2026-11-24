from pathlib import Path

import joblib


def save_model(model, path):
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path):
    model_path = Path(path)
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    return joblib.load(model_path)
