import streamlit as st
import pandas as pd
import pickle
from pytorch_tabnet.tab_model import TabNetRegressor
import random

# Load preprocessor and model
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

model = TabNetRegressor()
model.load_model("tabnet_model.zip")

# Lightweight text generation function
def generate_summary(rating, calories, fats, proteins, carbs, shelf_life, ingredient):
    summaries = [
        f"With a focus on balanced nutrition, the product, featuring {ingredient}, has earned a high rating of {rating:.2f}.",
        f"Low in fats ({fats} g) and high in proteins ({proteins} g), this product ensures a healthy dietary choice, achieving a commendable rating of {rating:.2f}.",
        f"The combination of {ingredient} and {carbs} g of carbs makes this product perfect for quick energy needs, reflected in its rating of {rating:.2f}.",
        f"This product with a shelf life of {shelf_life} days ensures lasting quality and nutrition, securing a rating of {rating:.2f}.",
        f"Based on its caloric value ({calories} kcal) and balanced macronutrients, this product excels in quality, earning a rating of {rating:.2f}."
    ]
    return random.choice(summaries)

# Streamlit app
st.title("Raw material quality")
st.markdown("Provide food details to predict the rating and get a summary.")

# Input fields
calories = st.number_input("Calories", min_value=0, max_value=1000, value=200)
fats = st.number_input("Fats (g)", min_value=0.0, max_value=50.0, value=10.0)
proteins = st.number_input("Proteins (g)", min_value=0.0, max_value=100.0, value=20.0)
carbs = st.number_input("Carbs (g)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
shelf_life = st.number_input("Shelf Life (days)", min_value=1, max_value=365, value=180)
ingredients = st.selectbox("Ingredients", [
    "Sugar", "Salt", "Flour", "Milk", "Eggs", "Butter", "Honey",
    "Chocolate", "Vanilla", "Cinnamon", "Yeast"
])

# Predict button
if st.button("Predict Rating"):
    # Prepare input data
    input_data = pd.DataFrame([{
        "calories": calories, "fats": fats, "proteins": proteins,
        "carbs": carbs, "temperature": temperature, "humidity": humidity,
        "shelf_life": shelf_life, "ingredients": ingredients
    }])

    # Preprocess input
    input_transformed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_transformed)
    rating = round(float(prediction[0]), 2)
    st.success(f"Predicted Rating: {rating}")

    # Generate and display recommendation summary
    summary = generate_summary(rating, calories, fats, proteins, carbs, shelf_life, ingredients)
    st.subheader("Recommendation Summary")
    st.text_area("Generated Summary", value=summary, height=200)