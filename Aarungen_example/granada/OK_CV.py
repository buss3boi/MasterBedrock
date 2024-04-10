# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:30:10 2024

@author: busse
"""

from Granada_plotting import wxvec, wyvec
import pandas as pd

#%% Import obs_xyz_gdf



import os
import sys

# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'DQM')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)


# Assuming OBS_XYZ is a DataFrame with columns 'X', 'Y', 'Z'
# Replace this with the actual column names in your dataset
X = OBS_XYZ_gdf[['Shape_X', 'Shape_Y']].values
y = OBS_XYZ_gdf['blengdber_'].values

#%% Kriging


import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pykrige.kriging_tools as kt

# Extracting X, Y, and Z values from OBS_XYZ
x = OBS_XYZ_gdf['Shape_X'].values
y = OBS_XYZ_gdf['Shape_Y'].values
z = OBS_XYZ_gdf['blengdber_'].values

# Plotting porportions
proportion = len(wxvec)/len(wyvec)

#%% How to plot. Kriging Spherical model, autoselect

OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='spherical'
    )

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

MSE = np.sqrt(np.mean(ss**2))
print(MSE)


# Plot the kriging result
plt.figure(figsize=(8*proportion, 8))
plt.scatter(x, y, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(wxvec, wyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("Spherical Variogram model")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

# RMSE on ss: 29


#%% Grid search params


import numpy as np
from sklearn.model_selection import GridSearchCV

from pykrige.rk import Krige

# 2D Kring param opt

param_dict = {
    "method": ["ordinary", "universal"],
    "variogram_model": ["linear", "power", "gaussian", "spherical", "exponential"],
    # "nlags": [4, 6, 8],
    # "weight": [True, False]
}

estimator = GridSearchCV(Krige(), param_dict, verbose=True, return_train_score=True)

# dummy data
XY = OBS_XYZ[['Shape_X', 'Shape_Y']].values

# run the gridsearch
estimator.fit(X=XY, y=z)


if hasattr(estimator, "best_score_"):
    print("best_score R² = {:.3f}".format(estimator.best_score_))
    print("best_params = ", estimator.best_params_)

print("\nCV results::")
if hasattr(estimator, "cv_results_"):
    for key in [
        "mean_test_score",
        "mean_train_score",
        "param_method",
        "param_variogram_model",
    ]:
        print(" - {} : {}".format(key, estimator.cv_results_[key]))



# 3D Kring param opt

param_dict3d = {
    "method": ["ordinary3d", "universal3d"],
    "variogram_model": ["linear", "power", "gaussian", "spherical"],
    # "nlags": [4, 6, 8],
    # "weight": [True, False]
}

estimator = GridSearchCV(Krige(), param_dict3d, verbose=True, return_train_score=True)

# dummy data
X3 = np.random.randint(0, 400, size=(100, 3)).astype(float)
y = 5 * np.random.rand(100)

# run the gridsearch
estimator.fit(X=X3, y=y)


if hasattr(estimator, "best_score_"):
    print("best_score R² = {:.3f}".format(estimator.best_score_))
    print("best_params = ", estimator.best_params_)

print("\nCV results::")
if hasattr(estimator, "cv_results_"):
    for key in [
        "mean_test_score",
        "mean_train_score",
        "param_method",
        "param_variogram_model",
    ]:
        print(" - {} : {}".format(key, estimator.cv_results_[key]))







