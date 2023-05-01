import numpy as np
import pandas as pd
# Importing libraries

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
 

from flask import Flask, render_template, request, jsonify
app = Flask(__name__, template_folder='/')
run_with_ngrok(app)   
  
@app.route("/")
def home():
    return render_template('index.html',data=["toto","tata"])
