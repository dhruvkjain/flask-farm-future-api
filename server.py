from flask import Flask, request, jsonify, render_template
from markupsafe import Markup
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from flask_cors import CORS
from fertilizer_py import fertilizer_dic

app = Flask(__name__)
CORS(app)  # Allows all origins by default

# Load the crop recommendation model
crop_recommendation_model = pickle.load(open('RandomForest.pkl', 'rb'))

# Handle crop prediction
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    try:
        # Get data from the JSON request
        data = request.get_json()

        # Check if all required fields are present
        required_fields = ['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields in the request'}), 400

        print(data)
        # Extract the data
        N = float(data['nitrogen'])
        P = float(data['phosphorous'])
        K = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Create an array for prediction
        prediction_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Perform the prediction
        prediction = crop_recommendation_model.predict(prediction_data)
        final_prediction = prediction[0]  # This should be a string like 'jute', 'rice', etc.

        return jsonify({'crop': final_prediction})
    
    except Exception as e:
        # Log the error
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Handle time series prediction for any crop
@app.route('/crop-price-predict', methods=['POST'])
def predict_crop_price():
    data = request.json
    state = data.get('state', 'Bihar')
    crop = data.get('crop', 'rice')
    
    # Construct the filename based on the crop name
    filename = f'{crop}.xlsx'
    try:
        df = pd.read_excel(filename, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        return jsonify({'error': f'File {filename} not found'}), 404
    
    # Build and fit the SARIMAX model
    model = sm.tsa.statespace.SARIMAX(df[state], order=(0, 1, 0), seasonal_order=(0,1,1,12))
    results = model.fit()
    
    # Make predictions
    pred = results.get_prediction(start=pd.to_datetime('2024-08-01'), end=pd.to_datetime('2024-12-01'), dynamic=False)
    pred_mean = pred.predicted_mean

    # Create a dictionary with date:price pairs
    pred_dict = {date.strftime("%d/%m/%Y"): price for date, price in pred_mean.items()}
    
    return jsonify(pred_dict)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    # Retrieve JSON data from request
    data = request.get_json()
    crop_name = str(data.get('cropname', ''))
    N = int(data.get('nitrogen', 0))
    P = int(data.get('phosphorous', 0))
    K = int(data.get('potassium', 0))

    df = pd.read_csv('fertilizer_csv.csv')

    # Search for the crop in the CSV file
    crop_data = df[df['Crop'].str.lower() == crop_name.lower()]

    if crop_data.empty:
        return jsonify({'error': 'Crop data not found'}), 404

    # Extract the recommended values of N, P, and K for the crop
    nr = crop_data['N'].values[0]
    pr = crop_data['P'].values[0]
    kr = crop_data['K'].values[0]

    # Calculate nutrient differences
    n = nr - N
    p = pr - P
    k = kr - K
    
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return jsonify({'recommendation': response})

if __name__ == '__main__':
    app.run(debug=True)
