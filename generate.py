import pandas as pd
import numpy as np

#size
length=1000
# Define ranges for numerical features
def generate_numerical_data():
    return {
        "calories": np.random.randint(50, 700, size=length),
        "fats": np.random.uniform(1, 30, size=length),
        "proteins": np.random.uniform(1, 50, size=length),
        "carbs": np.random.uniform(10, 100, size=length),
        "temperature": np.random.uniform(0, 50, size=length),
        "humidity": np.random.uniform(10, 90, size=length),
        "shelf_life": np.random.randint(1, 365, size=length),
    }

# Define a list of ingredients
ingredients = [
    "Sugar", "Salt", "Flour", "Milk", "Eggs", "Butter", "Honey",
    "Chocolate", "Vanilla", "Cinnamon", "Yeast"
]

# Generate categorical data
def generate_categorical_data():
    return np.random.choice(ingredients, size=length)

# Rules-based rating generation
def calculate_rating(row):
    score = 0
    # Increase score based on protein and decrease for high fat
    score += 1 if row["proteins"] > 30 else -1
    score -= 1 if row["fats"] > 20 else 0
    # Penalize extreme humidity and temperature
    if row["temperature"] > 40 or row["humidity"] > 80:
        score -= 2
    # Boost shelf life and penalize very short shelf life
    if row["shelf_life"] > 180:
        score += 2
    elif row["shelf_life"] < 30:
        score -= 2
    # Ensure rating is between 1 and 10
    return max(1, min(10, 5 + score))

# Generate the dataset
numerical_data = generate_numerical_data()
categorical_data = generate_categorical_data()

df = pd.DataFrame(numerical_data)
df["ingredients"] = categorical_data
df["rating"] = df.apply(calculate_rating, axis=1)

# Save to CSV
df.to_csv("food_dataset.csv", index=False)
print("Dataset generated and saved as food_dataset.csv")
