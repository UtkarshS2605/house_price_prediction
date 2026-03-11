import pickle
import numpy as np
from flask import Flask,request,app,jsonify,url_for,render_template

app = Flask(__name__)

#load the model
regmodel = pickle.load(open("regmodel.pkl","rb"))
scalar = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.get_json()['data']

    input_dict = {col:0 for col in columns}

    for key,value in data.items():
        if key in input_dict:
            input_dict[key] = value

    final_input = np.array(list(input_dict.values())).reshape(1,-1)

    scaled_input = scalar.transform(final_input)

    prediction = regmodel.predict(scaled_input)

    return jsonify(prediction[0])

if __name__ =="__main__":
    app.run(debug=True)
