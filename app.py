import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('rf_model.pkl', 'rb'))

region_df=pd.read_csv('region_df.csv')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        result = request.form

        longitude = result['longitude']
        latitude = result['latitude']
        lender_count = result['lender_count']
        popDensity = result['popDensity']
        precipitation = result['precipitation']
        region = result['region']

        idx=region_df[region_df['region']== region].index
        region=region_df.iloc[idx,1][0]
        
        inputs = np.array([[longitude,latitude,lender_count,popDensity,precipitation,region]]).reshape(1,-1)
        print(inputs.shape)

    prediction = model.predict(inputs)
    return render_template('index.html', Funded_amount ='Your funded amount is {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)      