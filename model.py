import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pytorch_tabnet.tab_model import TabNetRegressor
import pickle
import numpy as np

# Load dataset
df = pd.read_csv("food_dataset.csv")

# Separate features and target
X = df.drop("rating", axis=1)
y = df["rating"]

# Reshape target for TabNet
y = y.values.reshape(-1, 1)

# Define numeric and categorical features
numeric_features = ["calories", "fats", "proteins", "carbs", "temperature", "humidity", "shelf_life"]
categorical_features = ["ingredients"]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Convert preprocessed data to NumPy array (if necessary for TabNet)
X_train_transformed = np.array(X_train_transformed)
X_test_transformed = np.array(X_test_transformed)

# Train TabNet model
model = TabNetRegressor()
model.fit(
    X_train_transformed, y_train,
    eval_set=[(X_test_transformed, y_test)],
    eval_metric=["rmse"],
    max_epochs=100,
    patience=10,
    batch_size=128,
    virtual_batch_size=64
)

# Save model and preprocessor
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

model.save_model("tabnet_model")
print("Model and preprocessor saved.")
