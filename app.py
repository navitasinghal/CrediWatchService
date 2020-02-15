from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
api = Api(app) 
CORS(app)  

class Test(Resource):
    def get(self): 
        return jsonify({'message': 'health is Ok'}) 

def newData( data):
        new_data = dict((k.upper(),v) for k, v in data.items()) 
        new_data = dict((k.replace("_", " "),v) for k, v in new_data.items()) 
        return new_data

class ConsumerData(Resource):
    def get(self):
        return jsonify({'consumerdata' : 'consumer data'})

    
    def post(self):
        data = request.get_json()     # status code 
        new_data=(newData(data))
        x=(jsonify([new_data]))
        print(x, new_data)
        return jsonify([new_data])


api.add_resource(Test, '/') 
api.add_resource(ConsumerData, '/consumerData')
# driver function 
if __name__ == '__main__': 
    app.run(debug = True) 

