import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Helper function to choose a valid default for a category.
# It returns the first non-"0" value from a category list.
def get_valid_default(category_array):
    for cat in category_array:
        cat_str = str(cat).strip()
        if cat_str and cat_str != "0":
            return cat_str
    # If all values are "0" (unlikely), return "Unknown"
    return "Unknown"

# New helper function to validate an input value.
# If the value is "0" or not found in valid_list, return the first non-"0" valid category.
def validate_input_value(value, valid_list):
    value_str = str(value).strip()
    if value_str == "0" or value_str not in valid_list:
        # Return the first non-"0" value from valid_list.
        for cat in valid_list:
            cat_str = str(cat).strip()
            if cat_str != "0":
                return cat_str
        # Fallback, though this should not happen if valid_list is correct.
        return valid_list[0]
    return value_str

# Load trained model and encoder
with open("res_rec.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load dataset to get dynamic dropdown values for City and Cuisine
df = pd.read_csv("restaurants_updated.csv")

# Ensure dropdown values are strings and filter out "0" if any
unique_cities = [x for x in df["City"].astype(str).unique() if x.strip() != "0"]
unique_cuisines = [x for x in df["Cuisine"].astype(str).unique() if x.strip() != "0"]

# Retrieve the encoder's known categories for each categorical field as strings.
# Order in encoder.categories_: 0 = Location, 1 = Locality, 2 = City, 3 = Cuisine
location_categories = [str(cat).strip() for cat in encoder.categories_[0]]
locality_categories  = [str(cat).strip() for cat in encoder.categories_[1]]
city_categories      = [str(cat).strip() for cat in encoder.categories_[2]]
cuisine_categories   = [str(cat).strip() for cat in encoder.categories_[3]]

# Choose valid default values for Location and Locality using get_valid_default
default_location = get_valid_default(location_categories)
default_locality = get_valid_default(locality_categories)

# Debug prints (optional)
print("Location encoder categories:", location_categories)
print("Default Location:", default_location)
print("Locality encoder categories:", locality_categories)
print("Default Locality:", default_locality)
print("City encoder categories:", city_categories)
print("Cuisine encoder categories:", cuisine_categories)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", cities=unique_cities, cuisines=unique_cuisines)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get dropdown selections for City and Cuisine (as strings)
        city_input = request.form["city"]
        cuisine_input = request.form["cuisine"]

        # Validate City and Cuisine inputs using our helper function.
        city = validate_input_value(city_input, city_categories)
        cuisine = validate_input_value(cuisine_input, cuisine_categories)
        
        # For Location and Locality, we use the validated default values.
        loc_val = default_location
        locl_val = default_locality

        # Log the values for debugging
        print("Input values:")
        print("Location:", loc_val)
        print("Locality:", locl_val)
        print("City:", city)
        print("Cuisine:", cuisine)

        # Prepare input array with all 4 features in the order: Location, Locality, City, Cuisine
        categorical_input = np.array([[loc_val, locl_val, city, cuisine]])
        
        # Transform using the one-hot encoder
        categorical_encoded = encoder.transform(categorical_input)
        
        # Convert to DataFrame with proper column names from the encoder
        encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out())
        
        # Align columns to match the model's expected feature order.
        all_feature_names = model.feature_names_in_
        for missing_feature in set(all_feature_names) - set(encoded_df.columns):
            encoded_df[missing_feature] = 0  # Fill missing features with 0
        encoded_df = encoded_df[all_feature_names]
        input_data = encoded_df.to_numpy()
        
        # Debug prints (optional)
        print(f"Encoded Input Shape in Flask: {input_data.shape}")
        print(f"Final Input: {input_data}")
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        return render_template("index.html", 
                               cities=unique_cities, 
                               cuisines=unique_cuisines,
                               prediction_text=f"Predicted Popularity Rating: {prediction[0]:.2f}")
    except Exception as e:
        return render_template("index.html", 
                               cities=unique_cities, 
                               cuisines=unique_cuisines,
                               prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
