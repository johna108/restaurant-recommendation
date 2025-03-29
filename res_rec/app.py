import pickle
import joblib
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = "R.pkl"
with open(model_path, "rb") as f:
    model = joblib.load(f)  # Use joblib to avoid pickle issues

# Load restaurant data
csv_path = "restaurants_updated.csv"  # Ensure this file is in the same directory as app.py
restaurant_data = pd.read_csv(csv_path)

@app.route("/")
def index():
    cities = restaurant_data["City"].unique().tolist()
    cuisines = restaurant_data["Cuisine"].str.split(", ").explode().unique().tolist()
    return render_template("index.html", cities=cities, cuisines=cuisines)


@app.route("/predict", methods=["POST"])
def predict_restaurant():
    try:
        # Get user input from the form
        city = request.form["city"].strip().title()  # Normalize input (e.g., "delhi" -> "Delhi")
        cuisine = request.form["cuisine"].strip().title()

        # Filter restaurants based on City and Cuisine
        filtered_restaurants = restaurant_data[
            (restaurant_data["City"] == city) &
            (restaurant_data["Cuisine"].str.contains(cuisine, case=False, na=False))
        ]

        # Sort by Rating & Popularity Score
        top_restaurants = (
            filtered_restaurants.sort_values(by=["Rating", "popularity_score"], ascending=False)
            .head(5)  # Get top 5
        )

        # Extract restaurant names
        restaurant_names = top_restaurants["Name"].tolist()

        return render_template("index.html", result=restaurant_names)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True, port=8080)
