import sys
import pickle
import pathlib
import LabelEncoder as le
from joblib import load
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Response
from typing import List, Any

sys.path.append("../resources")

from df_config import df_labels

app=FastAPI()

class DocumentRequest(BaseModel):
    document: list[Any]

#model=load(pathlib.Path('model/random-forest-model.pkl'))
model=load(pathlib.Path(rf'C:\Users\AbdullahYousaf\OneDrive - Kubrick Group\Desktop\MLOps\MLOps_MedicalInsurance\model\random-forest-model.pkl'))

def df_input(X):
    df=pd.DataFrame(columns=df_labels)
    df.loc[len(df)] = X

    return df

@app.post('/prediction')
def transf_and_predict(request: DocumentRequest) -> Response:
    Xtest = request.document
    X = df_input(Xtest)

    # Transform input as per data pre-processing
    X = le.label_encode(X)
    X = le.feature_drop(X)
    X = le.polynomial_split(X)
    
    # Use the pre-processed input to predict output
    Y = model.predict(X)

    return Y

@app.get("/")
def root():
    return "Hello! This is just a homepage for the medical insurance app."

if __name__ == "__main__":
    uvicorn.run(app)


