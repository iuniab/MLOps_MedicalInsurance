"""
Scikit-Learn Pipelines Demo

"""

__date__ = "2023-06-08"
__author__ = "AbdullahYousaf"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from IPython.display import display

# %% --------------------------------------------------------------------------
# Set Random State
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)


# %% --------------------------------------------------------------------------
# Load Data and Train/Test Split
# -----------------------------------------------------------------------------

# Load the dataset
df = pd.read_csv(r'C:\Users\AbdullahYousaf\OneDrive - Kubrick Group\Desktop\MLOps\MLOps_MedicalInsurance\data\insurance.csv')

# Display the first few rows of the dataframe
print(df.head())

# Display the data types and missing values
print(df.info())

# Summary statistics for numerical features
print(df.describe())


# %% --------------------------------------------------------------------------
# Create pipeline
# -----------------------------------------------------------------------------
# The first argument in the pipeline is a list of tuples containing the stages
# Each tuple contains two elements
# The first element in the tuple is a name referring to the stage
# The second element is the estimator object

import seaborn as sns
import matplotlib.pyplot as plt

# Check for missing values
print(df.isnull().sum())

# Histograms for numerical features
df.hist(bins=15, figsize=(15, 10))
plt.show()

# Box plots for numerical features to identify outliers
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()



# %% --------------------------------------------------------------------------
# Fit and predict workflow on the pipeline
# -----------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# Assuming 'charges' is the target variable and it's included in the dataset
# Split the dataset into features and target variable
X = df.drop('charges', axis=1)  # or another column if 'charges' is not your target
y = df['charges']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Creating transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating a preprocessing and modeling pipeline
# Example with a linear regression model
from sklearn.linear_model import LinearRegression

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

# Fitting the model
pipeline.fit(X_train, y_train)

# Making predictions (as an example of how to use the pipeline)
predictions = pipeline.predict(X_test)

# The pipeline is now ready to be used for model training and predictions.
# It includes feature scaling for numerical features and one-hot encoding for categorical features.

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Making predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculating the metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²) score: {r2}")

# Interpretation:
# MAE provides an average error committed by the model's predictions.
# MSE is more sensitive to larger errors than MAE due to squaring the error terms.
# R² score represents the proportion of the variance for the dependent variable that's explained by the independent variables in the model.

# %%
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# Assuming df is your DataFrame after loading insurance.csv  # Adjust path as necessary

# Separating features and target
X = df.drop(columns=['charges'])  # Assuming 'charges' is the target variable
y = df['charges']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Creating the preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating the Random Forest pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(random_state=42))])

# Defining the parameter grid for RandomizedSearchCV
param_distributions = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_features': ['sqrt', 'log2'],
    'regressor__max_depth': [10, 20, 30, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True, False]
}

# Setting up the RandomizedSearchCV
rf_random_search = RandomizedSearchCV(rf_pipeline, param_distributions=param_distributions, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fitting the model
rf_random_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", rf_random_search.best_params_)

# Predicting and evaluating the model
y_pred = rf_random_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}, MSE: {mse}, R^2: {r2}")


# %%
