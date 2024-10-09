from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('../data/credit_risk_prob_model.pkl')

# Define a function for data preprocessing (if needed)
def preprocess_input(data):
    """
    This function can be extended to preprocess the incoming data
    like encoding categorical variables or scaling numerical ones.
    For now, it assumes that the incoming data is already preprocessed.
    """
    # Convert input data to DataFrame (Assuming JSON format)
    df = pd.DataFrame([data])
    return df

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json
        
        # Preprocess the data
        processed_data = preprocess_input(input_data)
        
        # Make predictions using the loaded model
        prediction = model.predict(processed_data)
        
        # Format the prediction as a JSON response
        response = {
            'prediction': int(prediction[0])  # Convert the NumPy int output to regular int
        }
        return jsonify(response)
    
    except Exception as e:
        # Return error message in case of failure
        return jsonify({'error': str(e)}), 400

# Home endpoint to check if API is working
@app.route('/')
def home():
    return "Credit Risk Model API is running!"

# Run the Flask app (for local development)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
