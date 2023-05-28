from flask import Flask, request, jsonify
import pickle
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "DO PHI SON"

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))
    day = int(request.form.get('day'))

    # Create input data for prediction
    input_data = np.array([[year, month, day]])
    # Create a new scaler for input data
    input_scaler = MinMaxScaler()
    input_scaled = input_scaler.fit_transform(input_data)
    #input_scaled = scaler.transform(input_data)

    # Reshape input data for LSTM model
    input_reshaped = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Make prediction
    prediction_scaled = model.predict(input_reshaped)
    prediction = scaler.inverse_transform(prediction_scaled)

    # Extract the predicted temperature
    temperature = prediction[0][0]

    return jsonify({'temperature': str(temperature)})

if __name__ == '__main__':
    app.run(debug=True)