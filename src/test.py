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


class DocumentRequest(BaseModel):
    document: list[Any]

def transf_and_predict(request: DocumentRequest) -> Response:
    test = request.document
    return test



test2 = {
    "input_one": {
        "document":
            [31, "male", 36.850, 0, "yes", "southwest"]
    },
    "output_one": {
        "results": {
            "count": 1,
            "predictions": [
                {
                    "charges": 15000,
                    "index": 1
                }
            ]
        }
    }
}

test = transf_and_predict(test2)
print(test)