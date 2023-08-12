import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Route for the home page, renders the input form
@app.route("/")
def home_page():
    return render_template('home.html')

# Route for handling form submission and prediction
@app.route("/", methods=['POST'])
def predict():
    # Extracting user inputs from the form
    Dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    # Loading the pre-trained machine learning model
    model = pickle.load(open('Model.sav', 'rb'))

    # Creating a DataFrame with user inputs
    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection,
             TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    # Identifying categorical features for encoding
    categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}

    # Encoding categorical features using LabelEncoder
    encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = encoder.fit_transform(df[feature])

    # Making a prediction using the loaded model
    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = probability * 100

    # Generating prediction messages based on the model output
    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    # Rendering the results along with the user input values
    return render_template("home.html", op1=op1, op2=op2,
                           Dependents=request.form['Dependents'],
                           tenure=request.form['tenure'],
                           OnlineSecurity=request.form['OnlineSecurity'],
                           OnlineBackup=request.form['OnlineBackup'],
                           DeviceProtection=request.form['DeviceProtection'],
                           TechSupport=request.form['TechSupport'],
                           Contract=request.form['Contract'],
                           PaperlessBilling=request.form['PaperlessBilling'],
                           MonthlyCharges=request.form['MonthlyCharges'],
                           TotalCharges=request.form['TotalCharges'])

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run()
