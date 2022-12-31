#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split
#from flask import Flask, flash, request, redirect, url_for, render_template


#app = Flask(__name__)

#@app.route('/')
#def hello_karios():
    
 #   return "hello world"

#@app.route('/display/<filename>')
#def display_image(filename):
 #   return redirect(url_for('static', filename=filename), code= 301)

#if __name__ == '__main__':
 #   app.run()
from flask import Flask,render_template,request,jsonify
#import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
@app.route('/api',methods=['POST'])
def predict():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    yPrediction=regressor.predict([[float(request.args['exp'])]])
       # np.array([float(request.args['experience'])]).reshape(1,1)
      #  )
    print("post")

    print(yPrediction)
    return "Salary should be "+ str(yPrediction[0])

@app.route('/api',methods=['GET'])
def predictAPI():
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    yPrediction=regressor.predict([[float(request.args['exp'])]])
    print("Get")
    print(yPrediction)

    return str(yPrediction)

if __name__ == '__main__':
   app.run()








