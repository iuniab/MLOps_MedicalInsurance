import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def data_split(X, Y):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 0)

    return X_train, X_test, Y_train, Y_test

def model_train(df):
    forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'friedman_mse',
                              random_state = 1,
                              n_jobs = -1)
    forest.fit(X_train,Y_train)
    
    return forest