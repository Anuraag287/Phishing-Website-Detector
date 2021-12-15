import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from extract import feat
app = Flask(__name__)
pred=pickle.load(open('XGClassifier.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    text=request.form.values()
    website=feat(text)
    final_features = [website]
    prediction = pred.predict(final_features)
    print(type(prediction[0]))
    prediction[0].astype(int)
    if(prediction[0]==1):
        output="Safe"
    else:
        output="Not Safe"
    return render_template('index.html', prediction_text='The requested website is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)