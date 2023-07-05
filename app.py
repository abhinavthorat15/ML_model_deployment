import pickle
from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import predictpipeline,CustomData


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('writing_score'),
            writing_score=request.form.get('reading_score')
        )
        pred_data = data.get_data_as_data_frame()
        pipeline = predictpipeline()
        prediction = pipeline.predict(pred_data)
        return render_template("home.html",results=prediction[0])
        #return jsonify({"The Prediction is ":prediction[0]})



if __name__=="__main__":
    app.run(host = '0.0.0.0',port=5000)
    
