from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features1=[int(x) for x in request.form.values()]
    final=[np.array(features1)]
    print(features1)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if(output==1):
        return render_template('index.html',pred='Your Loan will be approved')
    else:
        return render_template('index.html',pred="Your Loan will not be approved")


if __name__ == '__main__':
    app.run(debug=True)