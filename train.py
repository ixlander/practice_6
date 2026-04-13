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
