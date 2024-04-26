# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:52:26 2024

@author: busse
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']

"""
#%% Ensemble Random forrest regressor


# Define the model
model = RandomForestRegressor(random_state=42)

# Define cross-validation strategy; All models tested for = 42
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid to search
param_grid = {  
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Max depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Define scoring methods (MSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',  # Note: We use neg_mean_squared_error to indicate higher scores are better
    'R^2': 'r2'
}

# Set up grid search
RFR_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
RFR_grid.fit(X, y)

# Print best parameters and best scores
print("RFR Best Parameters: ", RFR_grid.best_params_)
print("Best MSE: ", -RFR_grid.best_score_)  # Negate the score to get MSE
print("Best R^2: ", RFR_grid.cv_results_['mean_test_R^2'][RFR_grid.best_index_])


# Best Parameters:  {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
# Best MSE:  15.631883574082952
# Best R^2:  0.6376508315034493
"""

from extended_cv import evaluate_model_ext_cv


# Example usage with your XGBoost setup:
model_type = RandomForestRegressor(random_state=42)
param_grid = {  
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Max depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

random_states = [12, 22, 32, 42, 52]

# mse_values, r2_values, opt_params = evaluate_model_ext_cv(model_type, param_grid, random_states, X, y)


# MSE Values: [16.122424334667095, 16.64564061648452, 17.55760725286112, 15.631883574082952, 15.614049805147374]
# R^2 Values: [0.622116397282514, 0.6055199804203806, 0.5926394877435796, 0.6376508315034493, 0.6467294653807188]
# Mean R^2: 0.6209312324661285
# Median R^2: 0.622116397282514
# Standard Deviation of R^2: 0.01992198754235969

"""
#%% Gradient Boosting


# Define the model
model = GradientBoostingRegressor(random_state=42)


# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.5],  # Learning rate
    'max_depth': [3, 5, 7],            # Max depth of the individual regression estimators
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Define scoring methods (MSE and R^2)
scoring = {
    'MSE': 'neg_mean_squared_error',  # Note: We use neg_mean_squared_error to indicate higher scores are better
    'R^2': 'r2'
}

# Set up grid search
GB_grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='MSE')

# Perform grid search
GB_grid.fit(X, y)

# Print best parameters and best scores
print("GB Best Parameters: ", GB_grid.best_params_)
print("Best MSE: ", -GB_grid.best_score_)  # Negate the score to get MSE
print("Best R^2: ", GB_grid.cv_results_['mean_test_R^2'][GB_grid.best_index_])

# GB Best Parameters:  {'learning_rate': 0.1, 'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
# Best MSE:  15.827176067644263
# Best R^2:  0.6363950580695261
"""

# Example usage with your XGBoost setup:
model_type = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.5],  # Learning rate
    'max_depth': [3, 5, 7],            # Max depth of the individual regression estimators
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

random_states = [12, 22, 32, 42, 52]

mse_values, r2_values, opt_params = evaluate_model_ext_cv(model_type, param_grid, random_states, X, y)


# MSE Values: [16.847558359856045, 17.197485666403463, 18.19847110915042, 15.827176067644263, 16.39622302816702]
# R^2 Values: [0.6082732616167114, 0.5972652362504894, 0.5728504516117775, 0.6363950580695261, 0.6329642971323307]
# Mean R^2: 0.6095496609361669
# Median R^2: 0.6082732616167114
# Standard Deviation of R^2: 0.023529933215215952