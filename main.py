from flask import *
import pickle
import numpy as np
import pandas as pd
import sklearn
import re
import random
from random import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open("iris_classification.pkl", "rb"))
    iris = loaded_model.predict(to_predict)
    return iris[0]


@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if (len(to_predict_list)==4):
            iris = ValuePredictor(to_predict_list)
            if iris >= int(0) and iris <= int(1):
                iris = 'Iris-Setosa'
            elif iris > int(1) and iris <= int(2):
                iris = 'Iris-Versicolor'
            elif iris > int(2) and iris <= int(3):
                iris = 'Iris-Virginica'
    return(render_template('index.html', prediction = "Species is {}".format(iris)))

if __name__ == "__main__":
    app.run(debug=True)