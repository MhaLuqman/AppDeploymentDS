import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl'))


@app.route('/')
def home():
    return render_template('/templates/index.html')



@app.route('/predict', methods=['POST'])
def predict():
    datat1 = request.form['age']
    datat2 = request.form['gender']
    datat3 = request.form['region']
    datat4 = request.form['occupation']
    datat5 = request.form['income']

    arr = np.array([[datat1, datat2, datat3, datat4, datat5]])
    predection = model.predict(arr)
    return render_template('/templates/predict.html', data=predection)


if __name__== "__main__":
    app.run(debug=True,port=5000)














