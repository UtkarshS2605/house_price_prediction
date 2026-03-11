import pickle
import numpy as np
from flask import Flask,request,app,jsonify,url_for,render_template

app = Flask(__name__)

#load the model
regmodel = pickle.load(open("regmodel.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
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

    scaled_input = scaler.transform(final_input)

    prediction = regmodel.predict(scaled_input)

    return jsonify(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():

    # get form values
    data = [float(x) for x in request.form.values()]

    # map form inputs
    input_dict = {
        "total_sqft": data[0],
        "bath": data[1],
        "balcony": data[2],
        "bhk": data[3]
    }

    # create full feature vector (319 columns)
    final_dict = {col: 0 for col in columns}

    for key in input_dict:
        if key in final_dict:
            final_dict[key] = input_dict[key]

    final_input = np.array(list(final_dict.values())).reshape(1,-1)

    scaled_input = scaler.transform(final_input)

    prediction = regmodel.predict(scaled_input)[0]

    return render_template(
        "home.html",
        prediction_text=f"Predicted House Price is {round(prediction,2)} Lakhs"
    )

if __name__ =="__main__":
    app.run(debug=True)
