
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Load the trained SVM model
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# These categories match your notebook's LabelEncoder setup.
# LabelEncoder sorts string categories alphabetically.
pclass_categories = ["First", "Second", "Third"]
gender_categories = ["Female", "Male"]
sibling_categories = ["One", "Three", "Two", "Zero"]
embarked_categories = ["Cherbourg", "Queenstown", "Southampton"]

# Input form
pclass = st.selectbox(
    "Passenger Class", ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender", ["Female", "Male"]
)

sibling = st.selectbox(
    "Number of Siblings", ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Southampton", "Queenstown"]
)

if st.button("Predict Survival"):
    # Encode inputs exactly as the notebook's LabelEncoders do.
    encoded_input = pd.DataFrame([{
        "PClass": pclass_categories.index(pclass),
        "Gender": gender_categories.index(gender),
        "Sibling": sibling_categories.index(sibling),
        "Embarked": embarked_categories.index(embarked)
    }])

    prediction = model.predict(encoded_input)[0]

    if prediction == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")

    st.caption(
        "This is an educational prediction from a trained ML model, "
        "not a guarantee of historical outcome."
    )
