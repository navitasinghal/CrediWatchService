from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


app = Flask(__name__)
api = Api(app)
CORS(app)


class Test(Resource):
    def get(self):
        return jsonify({'message': 'health is Ok'})


def newData(data):
    new_data = dict((k.upper(), v) for k, v in data.items())
    new_data = dict((k.replace("_", " "), v) for k, v in new_data.items())
    return new_data


class ConsumerData(Resource):
    def get(self):
        return jsonify({'consumerdata': 'consumer data'})

    def post(self):
        data = request.get_json()     # status code
        new_data = (newData(data))
        a = [new_data]

        model = joblib.load('model_new.pkl')
        test = pd.DataFrame(a)
        cols = ['STATE', 'COMPANY STATUS', 'CLASS', 'AUTHORIZED CAPITAL',
                'PAIDUP CAPITAL', 'ACTIVITY CODE', 'ACTIVITY DESCRIPTION']
        df_test = test[cols]
        cat = ['STATE', 'COMPANY STATUS', 'CLASS', 'ACTIVITY DESCRIPTION']
        for i in cat:
            le = preprocessing.LabelEncoder()
            df_test[i] = le.fit_transform(df_test[i].astype('str'))
        df_test = df_test.astype(str).astype(int)
        prediction = model.predict(df_test)
        if(prediction >= 0.5):
            result = "YES"
        else:
            result = "NO"
        return result


class Login(Resource):
    def get(self):
        return jsonify({'email':'navitasinghal77@gmail.com', 'password':'password'})
    def post(self):
        data = request.get_json()
        return jsonify({'email':'navitasinghal77@gmail.com', 'password':'password'})


class LoanRecommendation(Resource):
    def get(self):
        return jsonify({'Loan Recommendation' : 'Works well'})
    def post(self):
        data = request.get_json()
        x = [data]
        df = pd.DataFrame(x)
        print(x,df.head())
        res = {"result":[], "plotX":[],"plotY":[]}
        if(df['grade'][0]=='A' or df['grade'][0]=='B'):
            res["result"].append("Loans are in safe range")
        else:
            res["result"].append("Lower grades have higher incidence of defaults on loans")
        if(int(df['int_rate'][0])>=19):
            res["result"].append("More than 33 percent of loans are Charged Off.")
        else:
            res["result"].append("At rates of 19 percent and above, more than 33 percent of loans are Charged Off.")
        
        chance = 0.20* int(df['annual_inc'][0])
        if(int(df['annual_inc'][0])<chance):
            res["result"].append("defaults chance are low.")
        chance = 0.30* int(df['annual_inc'][0])
        if(int(df['annual_inc'][0])<chance):
            res["result"].append("defaults chance are high.")
        else:
            res["result"].append("defaults chance is low")
        res["plotX"].append(["B","A","C","D","E","F","G"])
        res["plotY"].append([3701,3170,2182,1484,883,356,92])
        return res


api.add_resource(Test, '/')
api.add_resource(ConsumerData, '/consumerData')
api.add_resource(Login, '/session')
api.add_resource(LoanRecommendation, '/loanRecommendation')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
