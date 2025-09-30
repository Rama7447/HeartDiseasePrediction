import streamlit as st
import pandas as pd
import pickle

st.title("Heart Disease Predictor")

# ----------------------------
# Model Names
# ----------------------------
modelnames = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']
algonames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']

# ----------------------------
# Prediction Function
# ----------------------------
def predict_heart_disease(data):
    predictions = []
    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(data)
        predictions.append(prediction[0])
    return predictions

# ----------------------------
# Streamlit Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# ----------------------------
# TAB 1: Single Prediction
# ----------------------------
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Up", "Flat", "Down"])

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [1 if fasting_bs == "> 120 mg/dl" else 0],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    if st.button("Submit"):
        st.subheader('Results')
        st.markdown('-------------------------')

        results = predict_heart_disease(input_data)

        for i, algo in enumerate(algonames):
            st.subheader(algo)
            if results[i] == 0:
                st.write("✅ No heart disease detected.")
            else:
                st.write("⚠️ Heart disease detected.")
            st.markdown('-------------------------------------')

# ----------------------------
# TAB 2: Bulk Prediction
# ----------------------------
with tab2:
    st.subheader("Upload CSV File for Bulk Prediction")
    st.info("""
    ⚠️ CSV file must have exactly these columns:
    ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
    'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)

            # validate columns
            expected_cols = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                             'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
            if all(col in input_data.columns for col in expected_cols):
                st.success("✅ File uploaded successfully.")

                # Run prediction using all models
                for i, modelname in enumerate(modelnames):
                    model = pickle.load(open(modelname, "rb"))
                    input_data[f"Prediction_{algonames[i]}"] = model.predict(input_data)

                st.subheader("Predictions:")
                st.write(input_data)

                # download link
                csv = input_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv,
                    file_name="heart_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Wrong columns in CSV file. Please fix headers and try again.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to start bulk prediction.")

# ----------------------------
# TAB 3: Model Information
# ----------------------------
with tab3:
    st.subheader("🧾 Model Information")
    st.markdown("ही app खालील 4 Machine Learning models वापरते:")

    model_info = {
        "Decision Tree": {
            "Description": "हे एक tree-structured model आहे. Features वर आधारित yes/no decisions घेऊन final prediction देते.",
            "Pros": "सोपं, interpret करायला easy, small datasets वर चांगलं काम करतं.",
            "Cons": "Overfitting होण्याची शक्यता जास्त.",
            "Accuracy": "उदा. ~80-85% (तुझ्या training वर अवलंबून)"
        },
        "Logistic Regression": {
            "Description": "हे एक statistical model आहे जे disease होण्याची probability calculate करतं.",
            "Pros": "Fast, simple, interpret करायला सोपं, probability outputs देते.",
            "Cons": "Linear decision boundary assumption असते, complex data साठी कमी useful.",
            "Accuracy": "उदा. ~82-86%"
        },
        "Random Forest": {
            "Description": "हे multiple decision trees एकत्र वापरून robust prediction करतं.",
            "Pros": "High accuracy, overfitting कमी, large datasets वर चांगलं काम करतं.",
            "Cons": "Interpret करायला कठीण, training थोडं slow.",
            "Accuracy": "उदा. ~85-90%"
        },
        "Support Vector Machine": {
            "Description": "हे best decision boundary शोधून दोन वर्ग वेगळे करतं.",
            "Pros": "High dimensional data वर चांगलं काम करतं.",
            "Cons": "Large datasets वर slow, tuning कठीण.",
            "Accuracy": "उदा. ~83-88%"
        }
    }

    for algo in algonames:
        st.markdown(f"### 🔹 {algo}")
        st.write("**Description:**", model_info[algo]["Description"])
        st.write("**Pros:**", model_info[algo]["Pros"])
        st.write("**Cons:**", model_info[algo]["Cons"])
        st.write("**Accuracy:**", model_info[algo]["Accuracy"])
        st.markdown("---")
