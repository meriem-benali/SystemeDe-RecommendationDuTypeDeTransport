from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')  # Adjust with your model file path

# Load saved encoders
encoder = joblib.load('encoder.pkl')  # Assuming it's for input encoding
label_encoder = joblib.load('label_encoder.pkl')  # Assuming it's for decoding output labels

print("Model and encoders loaded successfully.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        data = request.get_json()

        # Extract the features from the data
        features = data['features']

        # Convert the features to a NumPy array and reshape for prediction
        features = np.array(features).reshape(1, -1)  # Reshape to (1, N)

        # Make the prediction
        prediction = model.predict(features)

        # If prediction is probabilities (e.g., softmax), decode it to class label
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class index
            decoded_prediction = label_encoder.inverse_transform([predicted_class])[0]  # Decode to label
        else:
            decoded_prediction = prediction[0][0]  # For regression or binary classification

        # Return the prediction as a JSON response
        return jsonify({'prediction': str(decoded_prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
