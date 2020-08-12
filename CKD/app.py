import numpy as np
from feature_transform import transformAttributes
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('regressor', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    attributes = [float(x) for x in request.form.values()]
    print("Attributes entered : ",attributes) 
    std_x = transformAttributes(attributes)
    print("Transformed and scaled features : ",std_x)
    
    pred_class = model.predict(std_x)
    pred_prob = model.predict_proba(std_x)
    #output = round(prediction[0], 2)
    print("Predicted Class : ",pred_class[0])
    print("Predicted Probability : ",pred_prob[0])
    return render_template('index.html', prediction_text='CKD with %.2f probability'%(pred_prob[0][1]))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)