from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os

from tensorflow import keras

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Flask application

app = Flask(__name__)

# Model and pipeline paths

MODEL_PATH = "dhaka_house_model.keras"
PIPELINE_PATH = "dhaka_house_pipeline.pkl"

# Check files

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"{MODEL_PATH} not found. "
        "Run training.py first."
        
    )



if not os.path.exists(PIPELINE_PATH):

    raise FileNotFoundError(
        f"{PIPELINE_PATH} not found. "
        "Run training.py first."
    )

# Load model and preprocessing pipeline

model = keras.models.load_model(MODEL_PATH)

pipeline = joblib.load(PIPELINE_PATH)

# Home page

@app.route("/")
def home():
   return render_template("index.html")

# Prediction

@app.route("/predict",methods=["POST"])
def predict():

    try:

        location = request.form.get("location","").strip()

        area_text = request.form.get("area","").strip()

        bed_text = request.form.get("bed","").strip()

        bath_text = request.form.get("bath","").strip()

        # Validate location

        if not location:

            return render_template(
                "index.html",
                error="Please enter a location."
            )

        # Validate area

        try:

            area = float(area_text)

        except ValueError:

            return render_template(
                "index.html",
                error="Please enter a valid area."
            )


        if area <= 0:

            return render_template(
                "index.html",
                error="Area must be greater than 0."
            )

        # Validate bedrooms

        try:

            bed = float(bed_text)

        except ValueError:

            return render_template(
                "index.html",
                error="Please enter a valid number of bedrooms."
            )


        if bed <= 0:

            return render_template(
                "index.html",
                error="Bedrooms must be greater than 0."
            )

        # Validate bathrooms

        try:

            bath = float(bath_text)

        except ValueError:

            return render_template(
                "index.html",
                error="Please enter a valid number of bathrooms."
            )


        if bath <= 0:

            return render_template(
                "index.html",
                error="Bathrooms must be greater than 0."
            )


        # ------------------------------------------
        # Create input DataFrame
        # IMPORTANT:
        # These names must match training data
        # ------------------------------------------

        input_data = pd.DataFrame({

            "area": [area],

            "Bed": [bed],

            "Bath": [bath],

            "Location": [location]

        })


        print("\nInput data:")
        print(input_data)

        # Apply preprocessing

        prepared_data = pipeline.transform(input_data)

        # Make prediction

        prediction = model.predict(prepared_data,verbose=0)


        predicted_price = float(np.asarray(prediction).flatten()[0])

        # Display result
        print(
            f"\nPredicted price: {predicted_price:,.2f} Taka"
        )

        return render_template(

            "index.html",

            prediction=(
                f"{predicted_price:,.2f} Taka"
            )

        )


    except Exception as e:

        print("\nERROR:")
        print(e)


        return render_template(

            "index.html",

            error=(
                f"Prediction error: {str(e)}"
            )

        )


# Run application

if __name__ == "__main__":

    app.run(
        debug=True
    )