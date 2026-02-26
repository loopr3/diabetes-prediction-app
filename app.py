import streamlit as st
import kagglehub
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------- TITLE --------------------
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)
    return pd.read_csv(csv_path)

df = load_data()

# -------------------- TRAIN MODEL --------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------- USER INPUT --------------------
st.subheader("Patient Details")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# -------------------- PREDICTION --------------------
if st.button("Predict Diabetes"):

    new_patient = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
    new_patient = scaler.transform(new_patient)
    prediction = model.predict(new_patient)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
