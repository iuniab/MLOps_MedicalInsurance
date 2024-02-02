import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import pathlib
from joblib import dump

from transform import label_encode, feature_drop, polynomial_split
from train import data_split, model_train

df = pd.read_csv('./data/insurance.csv')

X = df.drop(columns=['charges'])
Y = df['charges']

X = label_encode(X)
X = feature_drop(X)
X = polynomial_split(X)

X_train, X_test, Y_train, Y_test = data_split(X, Y)
model=model_train(X_train, Y_train)

dump(model, pathlib.Path('model/random-forest-model.pkl'))


