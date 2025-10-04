from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for index page
@app.route("/")
def index():
    return render_template("index.html")

# Route for home page
@app.route("/home")
def home():
    return render_template('home.html')

# Route for prediction
@app.route("/predictdata/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            
            # Collect input values from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale input data and predict
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            # Render home page with the prediction result
            return render_template('home.html', result=result[0])
        except ValueError as e:
            return render_template('home.html', error ="Please enter valid numeric values for all fields.")
    else:
            return render_template("home.html")

if __name__ == "__main__":
        app.run(host="0.0.0.0")
