import sys
import pickle
import pathlib
import LabelEncoder as le
from joblib import load
import uvicorn

from fastapi import FastAPI

sys.path.append("../resources")

from df_config import df_labels

app=FastAPI()

model=load(pathlib.Path('model/random-forest-model.pkl'))

def df_input(X):
    a = X["input_one"]["document"]
    df=pd.DataFrame(columns=df_labels)
    df.loc[len(df1)]=a

    return df

@app.post('/prediction')
def transf_and_predict(X):
    X = df_input(X)

    # Transform input as per data pre-processing
    X = label_encode(X)
    X = feature_drop(X)
    X = polynomial_split(X)
    
    # Use the pre-processed input to predict output
    Y = model.predict(X)

    return Y

@app.get("/")
def root():
    logger.info("Access to '/' endpoint")
    return "Hello! This is just a homepage for the medical insurance app."

if __name__ == "__main__":
    uvicorn.run(app)


