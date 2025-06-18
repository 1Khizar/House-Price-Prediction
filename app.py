from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['GarageCars']),
            float(request.form['GarageArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['1stFlrSF']),
            float(request.form['ExterQual_TA']),
            float(request.form['KitchenQual_TA']),
            float(request.form['GarageFinish_Unf']),
        ]

        # Convert to array and reshape for model
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        formatted_prediction = f"{prediction:,.2f}"

        return render_template('result.html', predicted_price=formatted_prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
