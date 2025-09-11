import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# Load dataset
# ==============================
st.title("🩺 Diabetes Prediction Model")

df = pd.read_csv("diabetes.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Info")
st.write(df.describe())
st.write("Missing values:", df.isnull().sum())
st.write("Duplicated rows:", df.duplicated().sum())

# ==============================
# Exploratory Data Analysis (EDA)
# ==============================
st.subheader("Exploratory Data Analysis")

fig, ax = plt.subplots(figsize=(10, 10))
df.hist(ax=ax)
st.pyplot(fig)

st.write("Outcome counts")
st.write(df["Outcome"].value_counts())

fig, ax = plt.subplots()
sns.boxplot(data=df, x="Outcome", y="BMI", ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.histplot(data=df, x="Glucose", hue="Outcome", kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(data=df, x="Outcome", y="Age", ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.violinplot(data=df, x="Outcome", y="Pregnancies", ax=ax)
st.pyplot(fig)

st.write("Mean values by Outcome")
st.write(df.groupby("Outcome")[["Glucose", "BMI", "Age", "Pregnancies"]].mean())

# Correlation
corr = df.corr(numeric_only=True)["Outcome"].sort_values(ascending=False)
st.write("Correlation with Outcome")
st.write(corr)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ==============================
# Data Preprocessing
# ==============================
st.subheader("Data Preprocessing")

cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero_invalid:
    df[col].replace(0, np.nan, inplace=True)

df.fillna(df.median(), inplace=True)

st.write("After cleaning, missing values:", df.isnull().sum())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# Model Training and Evaluation
# ==============================
st.subheader("Model Training and Evaluation")

# Logistic Regression
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

st.write("### Logistic Regression")
st.write("Accuracy:", accuracy_score(y_test, y_pred_log))
st.text(classification_report(y_test, y_pred_log))

# SVM with GridSearch
params_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
svm = GridSearchCV(SVC(), params_svm, cv=3)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

st.write("### Support Vector Machine (SVM)")
st.write("Best params:", svm.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_svm))
st.text(classification_report(y_test, y_pred_svm))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

st.write("### Random Forest")
st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
st.text(classification_report(y_test, y_pred_rf))

# ==============================
# Prediction Function
# ==============================
st.subheader("Patient Prediction")

def predict_patient(model, patient_data, feature_names):
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)
    pred = model.predict(patient_scaled)
    return "Diabetic" if pred[0] == 1 else "Non-Diabetic"

feature_names = X.columns.tolist()

with st.form("prediction_form"):
    st.write("Enter patient data:")
    patient_data = []
    for col in feature_names:
        val = st.number_input(f"{col}", value=float(df[col].median()))
        patient_data.append(val)
    submitted = st.form_submit_button("Predict")

    if submitted:
        result = predict_patient(rf, patient_data, feature_names)
        st.success(f"Prediction: {result}")
