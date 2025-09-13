import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load model & encoders
model = joblib.load('Saved Models/best_model.pkl')
encoders = joblib.load('Saved Models/encoders.pkl')

st.set_page_config(page_title="Employee Salary Prediction App", layout="wide")

st.title("💼 Employee Salary Prediction App with Performance Dashboard")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar Input
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
fnlwgt = st.sidebar.number_input("fnlwgt", 10000, 1000000, 226802)
educational_num = st.sidebar.slider("Educational Num", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)

raw_input_df = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "educational-num": educational_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "gender": gender,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

st.write("### 🔎 Input Data (Displayed as Entered)")
st.write(raw_input_df)

# Encode Input
encoded_input = {col: encoders[col].transform([raw_input_df[col][0]])[0] if col in encoders else raw_input_df[col][0]
                 for col in raw_input_df.columns}
encoded_input_df = pd.DataFrame([encoded_input])

# Predict Single
if st.button("Predict Salary Class"):
    pred = model.predict(encoded_input_df)[0]
    proba = model.predict_proba(encoded_input_df)[0]
    st.markdown("---")
    st.write("### Encoded Input for Model")
    st.write(encoded_input_df)
    st.success(f"✅ Prediction: {'>50K' if pred == 1 else '≤50K'}")
    st.write(f"Prediction Probabilities — ≤50K: {proba[0]:.2f}, >50K: {proba[1]:.2f}")

# Batch Prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction with Performance Metrics")

uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type="csv")

if uploaded_file:
    raw_batch = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview", raw_batch.head())

    encoded_batch = raw_batch.copy()
    for col in encoders:
        if col in encoded_batch.columns:
            encoded_batch[col] = encoders[col].transform(encoded_batch[col])

    batch_preds = model.predict(encoded_batch)
    batch_proba = model.predict_proba(encoded_batch)
    encoded_batch['PredictedClass'] = batch_preds
    encoded_batch['Prob_<=50K'] = batch_proba[:, 0]
    encoded_batch['Prob_>50K'] = batch_proba[:, 1]

    merged = pd.concat([raw_batch.add_prefix('Raw_'), encoded_batch.add_prefix('Encoded_')], axis=1)
    st.write("✅ Batch Prediction Results Preview")
    st.write(merged.head())

    csv = merged.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Prediction CSV", csv, "predicted_results.csv", "text/csv")

    # If actual income column exists, show performance metrics
    if 'income' in raw_batch.columns:
        try:
            y_true = raw_batch['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
            y_pred = batch_preds

            st.markdown("### 📝 Model Performance on Uploaded Data")
            acc = accuracy_score(y_true, y_pred)
            st.write(f"**Accuracy:** {acc:.4f}")

            st.markdown("**Classification Report:**")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            st.markdown("**Confusion Matrix:**")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ Could not calculate metrics: {e}")

# Model Performance Dashboard Section
st.markdown("---")
st.markdown("## 📊 Data Visualizations")

try:
    sample_data = pd.read_csv('Dataset/adult 3.csv')
    st.write("📊 Sample Data Overview")
    st.dataframe(sample_data.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Age Distribution by Income")
        plt.figure(figsize=(5,3))
        sns.histplot(data=sample_data, x='age', hue='income', bins=30, kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("### Hours per Week vs Income")
        plt.figure(figsize=(5,3))
        sns.boxplot(data=sample_data, x='income', y='hours-per-week')
        st.pyplot(plt.gcf())
        plt.clf()

except Exception as e:
    st.warning(f"📂 Could not load sample dataset for visualization. Error: {e}")
