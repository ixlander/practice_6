import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Job_Status"
DROP_COLS = ["Employee_ID"]

NUM_COLS = [
    "Age", "Years_Experience", "Salary_Before_AI", "Salary_After_AI",
    "Work_Hours_Per_Week", "Job_Satisfaction", "Productivity_Change_%",
]
CAT_COLS = [
    "Gender", "Education_Level", "Industry", "Job_Role",
    "AI_Adoption_Level", "Automation_Risk", "Upskilling_Required", "Remote_Work",
]

def train(data_path: str = "data/ai_job_impact.csv"):
    print(f"loading dataset: {data_path}")
    df = pd.read_csv(data_path).drop(columns=DROP_COLS, errors="ignore")
    print(f"rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"target distribution:{df[TARGET].value_counts().to_string()}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    print("training RandomForest classifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("classification report:")
    print(classification_report(y_test, y_pred))

    artifact = {
        "pipeline": pipeline,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "classes": pipeline.classes_.tolist(),
    }
    joblib.dump(artifact, "model.joblib")
    print("model saved as model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai_job_impact.csv")
    args = parser.parse_args()
    train(args.data)