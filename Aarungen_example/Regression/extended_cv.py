# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:48:10 2024

@author: busse
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

def evaluate_model_ext_cv(model_type, param_grid, random_states, X, y):
    """
    Evaluates a machine learning regression model with different cross-validation random states.

    Args:
        model_type: The type of model to use (e.g., xgb.XGBRegressor).
        param_grid: A dictionary of hyperparameters to search.
        random_states: A list of random states to use for cross-validation.
        X: The input data.
        y: The target values.

    Returns:
        A tuple containing two lists:
            - mse_values: A list of MSE values for each random state.
            - r2_values: A list of R^2 values for each random state.
    """
    opt_params=[]
    mse_values = []
    r2_values = []

    for random_state in random_states:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        model = model_type  # Create a fresh model instance for each random state

        # Define scoring methods
        scoring = {
            'MSE': 'neg_mean_squared_error',
            'R^2': 'r2' 
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   scoring=scoring, cv=cv, refit='MSE')
        grid_search.fit(X, y)

        # Store the best results based on MSE
        opt_params.append(grid_search.best_params_)
        mse_values.append(-grid_search.best_score_)  # Using negative MSE
        r2_values.append(grid_search.cv_results_['mean_test_R^2'][grid_search.best_index_])  # Standard R^2
        
    print("MSE Values:", mse_values)
    print("R^2 Values:", r2_values)

    # Additional metrics
    mean_r2 = np.mean(r2_values)
    median_r2 = np.median(r2_values)
    std_r2 = np.std(r2_values)

    # Print the results
    print("Mean R^2:", mean_r2)
    print("Median R^2:", median_r2)
    print("Standard Deviation of R^2:", std_r2)

    return mse_values, r2_values, opt_params
