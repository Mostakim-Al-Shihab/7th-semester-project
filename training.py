import os
import numpy as np
import pandas as pd
import joblib

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# --------------------------------------------------
# TensorFlow settings
# --------------------------------------------------

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# File names
MODEL_FILE = "dhaka_house_model.keras"
PIPELINE_FILE = "dhaka_house_pipeline.pkl"
TEST_FILE = "test_dataset.csv"

# Build preprocessing pipeline


def build_pipeline(num_attributes, cat_attributes):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())])

    cat_pipeline = Pipeline(
        [("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

    combined_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)])

    return combined_pipeline


# Load dataset

data = pd.read_csv("houserentdhaka.csv")

# Clean Area

data["area"] = (
    data["Area"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("sqft", "", regex=False)
    .str.strip()
)

data["area"] = pd.to_numeric(data["area"], errors="coerce")

# Clean Price

def convert_price(price):

    price = str(price).strip()

    try:

        if "Thousand" in price:
            return float(price.replace("Thousand", "").strip()) * 1000

        elif "Lakh" in price:
            return float(price.replace("Lakh", "").strip()) * 100000

        elif "Crore" in price:
            return float(price.replace("Crore", "").strip() ) * 10000000

        else:
            # Try direct numeric value
            return float(price.replace(",", "").strip())

    except ValueError:
        return np.nan


data["price"] = data["Price"].apply(convert_price)


# Remove unnecessary columns

data = data.drop(["Area", "Price"],axis=1,errors="ignore")


# Remove invalid rows

data = data.dropna(subset=["area", "price", "Location"]).reset_index(drop=True)

# Create price categories for stratified split

data["Price_cat"] = pd.qcut(
    data["price"],
    q=5,
    labels=False,
    duplicates="drop")

# Train/Test split

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(data,data["Price_cat"]):

    train_data_set = data.iloc[train_index].drop("Price_cat",axis=1)

    test_data_set = data.iloc[ test_index ].drop("Price_cat",axis=1)
# Save test dataset
test_data_set.to_csv(TEST_FILE, index=False)
# Separate features and labels
housing_data = train_data_set.copy()
housing_labels= housing_data["price"].copy()
housing_features = housing_data.drop("price",axis=1)

# Numerical and categorical columns
housing_num = ["area","Bed","Bath"]
housing_cat = ["Location"]

# Check columns
for column in housing_num + housing_cat:

    if column not in housing_features.columns:

        raise ValueError(
            f"Column '{column}' not found in dataset. "
            f"Available columns: {housing_features.columns.tolist()}"
        )

# Build preprocessing pipeline

combined_pipeline = build_pipeline(housing_num,housing_cat)

# Transform training data

prepared_housing = combined_pipeline.fit_transform(housing_features)


# Build Deep Learning model

model = keras.Sequential([

    layers.Input(shape=(prepared_housing.shape[1],)),

    layers.Dense(128,activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(64,activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(32,activation="relu"),

    layers.Dense(1)
])
# Compile

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    # optimizer = "adam",
    loss="mean_squared_error",
    metrics=["mae","mse"]
)

#train

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights= True
)

model.fit(
    prepared_housing,
    housing_labels,
    epochs= 300,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = [early_stopping],
    verbose = 1

)
# Save model

model.save(MODEL_FILE)
joblib.dump(combined_pipeline,PIPELINE_FILE)

print("Model training completed.")
print("Model saved:", MODEL_FILE)
print("Pipeline saved:", PIPELINE_FILE)

# Test predictions

prepared_test = combined_pipeline.transform(test_data_set.drop("price", axis=1))
predictions = model.predict(prepared_test,verbose=0).flatten()

# Save predictions

prediction_output = test_data_set.drop("price", axis=1).copy()

prediction_output["actual_price"] = test_data_set["price"].values

prediction_output["predicted_price"] = predictions

prediction_output.to_csv("DL_test_dataset_with_predictions.csv",index=False)


print(
    "\nPredictions saved to DL_test_dataset_with_predictions.csv")