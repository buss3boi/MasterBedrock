# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:35:49 2024

@author: busse
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Load your dataset
OBS_XYZ_gdf = pd.read_csv("OBS_XYZ_gdf.csv")  # Assuming it's a CSV file

# Define features (X) and target variable (y)
X = OBS_XYZ_gdf[['X', 'Y']]
y = OBS_XYZ_gdf['Z']

# Generate a meshgrid of X and Y values
x_min, x_max = X['X'].min(), X['X'].max()
y_min, y_max = X['Y'].min(), X['Y'].max()
x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 100)
X_mesh, Y_mesh = np.meshgrid(x_values, y_values)

#%% Random Forest regressor prediction surface

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest regressor model
rf_reg = RandomForestRegressor(random_state=42, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=200)  # You can adjust the number of estimators as needed
rf_reg.fit(X, y)

# Predict Z values for the meshgrid using the Random Forest regressor
Z_pred_rf = rf_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_rf = Z_pred_rf.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_rf, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from Random Forest Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)
plt.savefig('random_forest_pred_surface.png')

plt.show()


#%% Gradient Boosting regressor prediction surface

from sklearn.ensemble import GradientBoostingRegressor

# Train a Gradient Boosting regressor model
gb_reg = GradientBoostingRegressor(random_state=42, learning_rate=0.1, max_depth=7, min_samples_leaf=4, min_samples_split= 2, n_estimators=200)  # You can adjust the number of estimators and learning rate as needed
gb_reg.fit(X, y)

# Predict Z values for the meshgrid using the Gradient Boosting regressor
Z_pred_gb = gb_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_gb = Z_pred_gb.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_gb, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from Gradient Boosting Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)
plt.savefig('gradient_boosting_pred_surface.png')

plt.show()


#%% XGBoost regressor prediction surface

import xgboost as xgb

# Train an XGBoost regressor model
xgb_reg = xgb.XGBRegressor(random_state=42, colsample_bytree=1.0, learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.6)  # You can adjust the number of estimators and learning rate as needed
xgb_reg.fit(X, y)

# Predict Z values for the meshgrid using the XGBoost regressor
Z_pred_xgb = xgb_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_xgb = Z_pred_xgb.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_xgb, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from XGBoost Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)
plt.savefig('xgboost_pred_surface.png')

plt.show()


