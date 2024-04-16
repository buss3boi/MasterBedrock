# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:46:53 2024

@author: busse
"""

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykrige.ok import OrdinaryKriging



from Granada_plotting import wxvec, wyvec, dx, dy


# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'Regression')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)


#%% Plot prediciton surface

# Extracting X, Y, and Z values from OBS_XYZ
x = OBS_XYZ_gdf['X'].values
y = OBS_XYZ_gdf['Y'].values
z = OBS_XYZ_gdf['Z'].values


# Exponential auto, increased nugget = {'range': 2275, 'sill': 43, 'nugget': 5}
# Exponential curve_fit auto increased nugget variogram_parameters = {'range': 1826.09, 'sill':46, 'nugget': 7}



# Variogram with custom parameters
variogram_parameters = {'range': 2275, 'sill': 43, 'nugget': 9.2e-10}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Kriging Prediction Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_pred_surface.png')
plt.show()



#%% Curve fit prediction surface

# Extracting X, Y, and Z values from OBS_XYZ
x = OBS_XYZ_gdf['X'].values
y = OBS_XYZ_gdf['Y'].values
z = OBS_XYZ_gdf['Z'].values


# Exponential auto, increased nugget = {'range': 2275, 'sill': 43, 'nugget': 5}
# Exponential curve_fit auto increased nugget variogram_parameters = {'range': 1826.09, 'sill':46, 'nugget': 7}



# Variogram with custom parameters
variogram_parameters = {'range': 1826.09, 'sill':46, 'nugget': 10e-10}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Kriging Prediction Surface skgstat')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_pred_surface_skg.png')
plt.show()


#%% Kriging increased nugget prediction surface


# Variogram with custom parameters
variogram_parameters = {'range': 2275, 'sill': 43, 'nugget': 5}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Kriging Prediction Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_pred_surface_nug.png')
plt.show()


#%% Kriging pykrige scaled data


# Variogram with custom parameters
variogram_parameters = {'range': 5661, 'sill': 50, 'nugget': 1}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    variogram_parameters=variogram_parameters,
    anisotropy_angle=-45,
    anisotropy_scaling=2.2,
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Kriging Prediction Surface Pykrige Scaled')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_pred_surface_pykrige_scaled.png')
plt.show()