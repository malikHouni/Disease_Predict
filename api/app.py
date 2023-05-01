import numpy as np
import pandas as pd
# Importing libraries
from scipy.stats import mode
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")


# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder_classes_
}
 

from flask import Flask, render_template, request, jsonify
app = Flask(__name__, template_folder='/')
run_with_ngrok(app)   
  
@app.route("/")
def home():
    return render_template('index.html',data=["toto","tata"])
