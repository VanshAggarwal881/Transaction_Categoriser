# to load trained model

import joblib
from pathlib import Path

# a path object pointing to the model file saved during training.
MODEL_PATH = Path("models/transaction_model.joblib")

model = None

def load_model():
    global model    

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    return model

def predict_category(text: str):
    """
    Takes a raw transaction description and predicts:
    - category
    - confidence score
    """
    model = load_model()
    
    predicted_label = model.predict([text])[0]
    
    probabilities = model.predict_proba([text])[0] # [0.8, 0.1, 0.1]
    
    label_index = list(model.classes_).index(predicted_label)
    confidence = probabilities[label_index]
    
    return {
        "text": text,
        "category": predicted_label,
        "confidence": round(float(confidence), 3)
    }

if __name__ == "__main__":
    sample = "Starbucks coffee payment"
    result = predict_category(sample)
    print(result)
