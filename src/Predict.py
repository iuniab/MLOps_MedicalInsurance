import pickle
import LabelEncoder as le

from fastapi import FastAPI

app=FastAPI()

model=pickle.load(open('../model/random-forest-model.pkl','wb'))

@app.post('/prediction')
def transf_and_predict(X):
    # Transform input as per data pre-processing
    X = label_encode(X)
    X = feature_drop(X)
    X = polynomial_split(X)
    
    # Use the pre-processed input to predict output
    Y = model.predict(X)

    return Y



