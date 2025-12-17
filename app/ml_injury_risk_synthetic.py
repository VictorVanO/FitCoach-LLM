"""
ML MODEL – Injury Risk Prediction (Synthetic Dataset)
----------------------------------------------------
Target dataset structure:
Player_Age, Player_Weight, Player_Height, Previous_Injuries,
Training_Intensity, Recovery_Time, Likelihood_of_Injury

Target:
- Likelihood_of_Injury (0 = low risk, 1 = high risk)

Model choice:
- Logistic Regression (interpretable, lightweight, jury-friendly)
- Non-LLM ML model → constraint respected
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

MODEL_PATH = "models/injury_risk_synthetic.joblib"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from raw dataset."""
    df = df.copy()

    # Height in meters
    df["height_m"] = df["Player_Height"] / 100

    # BMI calculation
    df["bmi"] = df["Player_Weight"] / (df["height_m"] ** 2)

    features = pd.DataFrame({
        "age": df["Player_Age"],
        "bmi": df["bmi"],
        "training_intensity": df["Training_Intensity"],
        "recovery_time": df["Recovery_Time"],
        "previous_injuries": df["Previous_Injuries"],
    })

    return features


def train_and_save_model(csv_path: str):
    """Train the injury risk model and save it."""
    df = pd.read_csv(csv_path)

    X = prepare_features(df)
    y = df["Likelihood_of_Injury"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nInjury Risk Model – Evaluation")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")



def load_model():
    """Load trained model."""
    return joblib.load(MODEL_PATH)


def predict_injury_risk(
    model,
    age: int,
    weight: float,
    height_cm: float,
    training_intensity: float,
    recovery_time: int,
    previous_injuries: int,
):
    """Predict injury risk probability."""
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    X = pd.DataFrame([
        {
            "age": age,
            "bmi": bmi,
            "training_intensity": training_intensity,
            "recovery_time": recovery_time,
            "previous_injuries": previous_injuries,
        }
    ])

    proba = model.predict_proba(X)[0][1]
    label = "High" if proba >= 0.5 else "Low"

    return {
        "risk_label": label,
        "risk_probability": round(float(proba), 3),
        "bmi": round(float(bmi), 2),
    }
