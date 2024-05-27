from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
loaded_model = pickle.load(open('medical_insurance_cost_predictor.sav', 'rb'))

# Define a route to render the HTML form
@app.route('/')
def home():
    return render_template('medical_insurance_form.html')

# Define a route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker = float(request.form['smoker'])
        region = float(request.form['region'])

        # Make a prediction
        input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
        prediction = loaded_model.predict(input_data)

        return render_template('medical_insurance_result.html', prediction=prediction[0])
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)