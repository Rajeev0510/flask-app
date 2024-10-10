
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model using the absolute path
model_path = '/Users/rajeevkumar/Documents/Gene Editing_ML/models/cd19_model.joblib'
scaler_path = '/Users/rajeevkumar/Documents/Gene Editing_ML/models/scaler_cd19_model.joblib'
kmeans_path = '/Users/rajeevkumar/Documents/Gene Editing_ML/models/kmeans_cd19_model.joblib'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
kmeans = joblib.load(kmeans_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    viability_percent = float(request.form['viability_percent'])
    cd4_cd8_ratio = float(request.form['cd4_cd8_ratio'])
    exhaustion_marker = request.form['exhaustion_marker']
    persistence_days = float(request.form['persistence_days'])
    
    # Map exhaustion marker to numerical values
    exhaustion_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    exhaustion_marker_num = exhaustion_mapping[exhaustion_marker]
    
    # Prepare the feature array
    features = np.array([[viability_percent, cd4_cd8_ratio, exhaustion_marker_num, persistence_days]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict the cluster using KMeans
    cluster = kmeans.predict(features_scaled)
    
    # Add the cluster information
    final_features = np.append(features_scaled, cluster).reshape(1, -1)
    
    # Get the prediction
    prediction = model.predict(final_features)
    
    # Map prediction to a readable output
    prediction_text = 'Remission' if prediction[0] == 1 else 'No Remission'
    
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

