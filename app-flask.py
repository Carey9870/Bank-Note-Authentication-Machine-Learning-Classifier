from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

# instantiation
app = Flask(__name__)
Swagger(app)

# load model file
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All......'

@app.route('/predict', methods=['GET'])
def predict_note_authentication():
    
    """
    Let's Authenticate the Banks Note
    This is using docstrings for specification
    
    ---
    parameters:
    
        -   name: variance 
            in: query
            type: number
            required: true
        -   name: skewness
            in: query
            type: number
            required: true
        -   name: curtosis
            in: query
            type: number
            required: true
        -   name: entropy
            in: query
            type: number
            required: true
    
    responses:
        200:
            description: The output values
    
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted values are" + str(prediction)

# run
if __name__ == '__main__':
    app.run(debug=True)