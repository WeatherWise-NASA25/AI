from flask import Flask, request, jsonify
import pandas as pd
import joblib
import torch

# Import the necessary components from your script
from weather_prediction import load_and_merge_data, preprocess_data, predict_weather, MLP

app = Flask(__name__)

# --- Global Variables for Model and Data ---
# These will be loaded once when the app starts
full_df = None
preprocessor = None
mlp_model = None

def load_resources():
    """Load all the necessary data and models into memory."""
    global full_df, preprocessor, mlp_model

    print("Loading data and models...")
    try:
        # 1. Load and preprocess the dataset
        full_df = load_and_merge_data()
        full_df = preprocess_data(full_df)

        # 2. Load the preprocessor
        preprocessor = joblib.load('preprocessor.joblib')

        # 3. Load the MLP model
        # We need to know the input size to initialize the model before loading the state dict
        # Create a dummy input to determine the size after preprocessing
        dummy_input_df = full_df.head(1)[preprocessor.feature_names_in_]
        input_size = preprocessor.transform(dummy_input_df).shape[1]
        target_columns = ['temperature_c', 'humidity_percent', 'precipitation_mm']

        mlp_model = MLP(input_size, len(target_columns))
        mlp_model.load_state_dict(torch.load('mlp_weather_model.pth'))
        mlp_model.eval() # Set model to evaluation mode

        print("Data and models loaded successfully.")

    except Exception as e:
        print(f"Error loading resources: {e}")
        # Depending on the desired behavior, you might want to exit the app
        # For now, we'll let it run, but endpoints will fail.

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Endpoint to get a weather prediction."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON body'}), 400

    city = data.get('city')
    date_str = data.get('date')

    if not city or not date_str:
        return jsonify({'error': 'Missing required parameters in JSON body: city and date'}), 400

    if full_df is None or preprocessor is None or mlp_model is None:
        return jsonify({'error': 'Server resources not loaded. Please check server logs.'}), 500

    try:
        # The predict_weather function from your script can be used directly
        # Note: We pass the loaded global variables to it.
        results = predict_weather(city=city, date_str=date_str, full_df=full_df, export_to_csv=False)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

if __name__ == '__main__':
    # Load all resources before starting the server
    load_resources()
    # Run the Flask app
    app.run(host='0.0.0.0', debug=True, port=5000)
