# Backend Flask for heart disease detector
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('heart_page.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    output = prediction[0]

    if output == 0:
        return render_template('heart_page.html', prediction_text='No Heart Disease Detected: {}'.format(output))

    elif output == 1:
        return render_template('heart_page.html', prediction_text=' Heart Disease Detected: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
