import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path


def load_data(csv_path = "data/transactions.csv"):
    return pd.read_csv(csv_path)

def train_model():
    """Train classifier using TF-IDF + Logistic Regression."""

    # TRAIN MODEL
    # df = Pandas DataFrame = looks like an Excel sheet inside Python.
    
    df = load_data()

    # COLUMNS
    X = df["description"]
    y = df["category"]

    # train_test_split randomly shuffles data before splitting.
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

    # ML pipeline
    # Input text → TF-IDF → Model → Prediction
    # ML pipeline
    model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=200))
    ])


    # Train
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    if not isinstance(report, str):
        import pprint
        report = pprint.pformat(report)
    matrix = confusion_matrix(y_test, y_pred)

    print("\n==== Classification Report ====")
    print(report)

    print("\n==== Confusion Matrix ====")
    print(matrix)

    # Save metrics to file
    metrics_path = Path("metrics_report.txt")
    with open(metrics_path, "w") as f:
        f.write("==== Classification Report ====" + "\n")
        f.write(report + "\n\n")
        f.write("==== Confusion Matrix ====" + "\n")
        f.write(str(matrix) + "\n")
    print(f"\nMetrics report saved -> {metrics_path}")

    # save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model, models_dir / "transaction_model.joblib")
    print("\nModel saved -> models/transaction_model.joblib")

if __name__ == "__main__":
    train_model()
