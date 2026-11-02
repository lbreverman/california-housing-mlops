from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from ollama import chat


# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="California Housing AI Assistant",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Housing AI Assistant")

st.write(
    "Estimate median housing values in California "
    "neighborhoods using a trained machine learning model."
)


# --------------------------------------------------
# Load the Best Random Forest Model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# --------------------------------------------------
# Store Predictions for the AI Assistant
# --------------------------------------------------

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "latest_inputs" not in st.session_state:
    st.session_state.latest_inputs = None


# --------------------------------------------------
# Neighborhood Information
# --------------------------------------------------

st.subheader("Enter Neighborhood Information")

med_inc = st.number_input(
    "Median income (in $10,000 units)",
    min_value=0.0,
    value=6.5
)

house_age = st.number_input(
    "Median house age",
    min_value=0.0,
    value=25.0
)

ave_rooms = st.number_input(
    "Average rooms per household",
    min_value=0.0,
    value=5.0
)

ave_bedrms = st.number_input(
    "Average bedrooms per household",
    min_value=0.0,
    value=1.0
)

population = st.number_input(
    "Block-group population",
    min_value=0.0,
    value=1200.0
)

ave_occup = st.number_input(
    "Average household occupancy",
    min_value=0.0,
    value=3.0
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=33.7
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-116.4
)


# --------------------------------------------------
# Determine Income Category
# --------------------------------------------------

if med_inc <= 2:
    income_category = "Low"
elif med_inc <= 4:
    income_category = "Medium"
elif med_inc <= 6:
    income_category = "High"
else:
    income_category = "Very High"


# --------------------------------------------------
# Predict Housing Value
# --------------------------------------------------

if st.button("Predict Housing Value"):

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
        "IncomeCategory": income_category
    }])

    try:
        prediction = model.predict(input_data)[0]
        estimated_value = prediction * 100000

        # Save the prediction and neighborhood inputs
        st.session_state.latest_prediction = float(
            estimated_value
        )

        st.session_state.latest_inputs = (
            input_data.iloc[0].to_dict()
        )

        st.success(
            f"Estimated Median House Value: "
            f"${estimated_value:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate of "
            "neighborhood median house value, not an "
            "individual property appraisal."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Local AI Assistant Powered by Llama 3.2
# --------------------------------------------------

st.divider()

st.subheader("🤖 Ask the AI Assistant")

st.write(
    "Ask questions about California housing prices, "
    "machine learning, or the factors that influence "
    "neighborhood housing values."
)

user_question = st.text_area(
    "Your question",
    placeholder="What factors influence California housing prices?"
)

# Provide the latest Random Forest prediction to Llama
if st.session_state.latest_prediction is not None:
    prediction_context = (
        f"The Random Forest model predicted a neighborhood "
        f"median house value of "
        f"${st.session_state.latest_prediction:,.2f}.\n"
        f"Neighborhood inputs: "
        f"{st.session_state.latest_inputs}\n"
        "These are the actual inputs and prediction from the "
        "application. Explain possible relationships, but do "
        "not claim to know exact feature contributions."
    )
else:
    prediction_context = (
        "No housing prediction has been generated yet. "
        "If the user asks about their prediction, instruct "
        "them to click Predict Housing Value first."
    )

if st.button("Ask AI Assistant"):

    if not user_question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Llama 3.2 is thinking..."):

            try:
                response = chat(
                    model="llama3.2:3b",
                    messages=[
                        {
    "role": "system",
    "content": (
        "You are a helpful AI assistant for a California "
        "housing price prediction application. "
        "Explain housing trends and machine learning "
        "in clear, beginner-friendly language. "
        "The application uses a trained Random Forest model "
        "to predict neighborhood median house values. "
        "Use the following application context when relevant:\n"
        f"{prediction_context}\n"
        "Never invent housing predictions or model metrics. "
        "Do not claim that you calculated feature importance "
        "or exact reasons for a prediction. "
        "If a prediction exists, use its actual value. "
        "If no prediction exists, direct the user to the "
        "Predict Housing Value button."
    )
},

                        {
                            "role": "user",
                            "content": user_question
                        }
                    ]
                )

                st.markdown("### AI Assistant Response")
                st.write(response.message.content)

            except Exception as error:
                st.error(
                    f"Unable to connect to the local AI model: {error}"
                )