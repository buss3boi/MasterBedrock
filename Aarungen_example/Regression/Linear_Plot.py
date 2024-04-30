# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:34:13 2024

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

#%% Linear regression prediction surface


# Train a linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Predict Z values for the meshgrid
Z_mesh_linear = linear_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_linear = Z_mesh_linear.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_linear, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from Linear Regression')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=30, azim=-130)
plt.savefig('linear_regression_pred_surface.png')
plt.show()


#%% Decision tree prediction surface

from sklearn.tree import DecisionTreeRegressor

# Train a decision tree regressor model
tree_reg = DecisionTreeRegressor(random_state=42, max_depth= 20, min_samples_leaf= 4, min_samples_split= 2)
tree_reg.fit(X, y)

# Predict Z values for the meshgrid using the decision tree regressor
Z_mesh_tree = tree_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_tree = Z_mesh_tree.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_tree, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from Decision Tree Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=30, azim=-70)
plt.savefig('decision_tree_pred_surface.png')

plt.show()


#%% SVR prediction surface

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an SVR model
svr = SVR(kernel='rbf', C=10000, gamma='auto')
svr.fit(X_scaled, y)

# Predict Z values for the meshgrid using the SVR model
X_mesh_scaled = scaler.transform(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))
Z_mesh_svr = svr.predict(X_mesh_scaled)
Z_mesh_svr = Z_mesh_svr.reshape(X_mesh.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_svr, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from Support Vector Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(azim=-120)
plt.savefig('SVR_pred_surface.png')

plt.show()


#%% KNN prediction surface

from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata

# Train a KNN regressor model
knn_reg = KNeighborsRegressor(n_neighbors=3, p=1, weights='distance')  # You can adjust the number of neighbors as needed
knn_reg.fit(X, y)

# Predict Z values for the meshgrid using the KNN regressor
Z_pred_knn = knn_reg.predict(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())))

# Interpolate the predicted Z values to create a smooth surface
Z_mesh_knn = griddata((X_mesh.ravel(), Y_mesh.ravel()), Z_pred_knn, (X_mesh, Y_mesh), method='cubic')

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh_knn, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Predicted Surface from K-Nearest Neighbors Regressor')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=40, azim=-55)
plt.savefig('KNN_pred_surface.png')
# ax.view_init(elev=45, azim=-120)


plt.show()




