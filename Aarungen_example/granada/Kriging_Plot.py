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


# Load dem convolution

dem_conv_krige_path = os.path.join(DQM_path, 'dem_convoluted_kriging.csv')

# Read OBS_XYZ.csv into a DataFrame
DEM_convolution = pd.read_csv(dem_conv_krige_path)


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

vari = np.mean(abs(ss))
print('Kriging Variance: {}'.format(vari))

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


# Plot the 3D kriging residuals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z variance')
ax.set_title('Kriging Variance Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_var_surface.png')
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

vari = np.mean(abs(ss))
print('Kriging Variance skg: {}'.format(vari))

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


# Plot the 3D kriging residuals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z variance')
ax.set_title('Kriging Variance Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_var_surface_skg.png')
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

vari = np.mean(abs(ss))
print('Kriging Variance Inc Nug: {}'.format(vari))

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


# Plot the 3D kriging residuals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z variance')
ax.set_title('Kriging Variance Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_var_surface_nug.png')
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
z_pred_pkr, ss = OK.execute('grid', wxvec, wyvec)

vari = np.mean(abs(ss))
print('Kriging Variance pykrige scaled: {}'.format(vari))

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred_pkr, cmap='viridis')

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


# Plot the 3D kriging residuals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z variance')
ax.set_title('Kriging Variance Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_var_surface_pykrige_scaled.png')
plt.show()





#%% Kriging pre-scaled data

# from Kriging_manual import fwxvec, fwyvec, final_data

# fx = final_data[:, 0]
# fy = final_data[:, 1]


# # Variogram with custom parameters
# variogram_parameters = {'range': 2573, 'sill': 50, 'nugget': 4}

# # Create the kriging object
# OK = OrdinaryKriging(
#     fx,
#     fy,
#     z,
#     variogram_model='exponential',
#     verbose=True,
#     enable_plotting=True,
#     variogram_parameters=variogram_parameters
# )

# # Perform the kriging interpolation
# z_pred, ss = OK.execute('grid', fwxvec, fwyvec)

# vari = np.mean(abs(ss))
# print('Kriging Variance prescaled data: {}'.format(vari))

# # Assuming you have already calculated z_pred and created wxvec, wyvec

# # Create meshgrid from wxvec and wyvec
# wx, wy = np.meshgrid(fwxvec, fwyvec)

# # Plot the 3D surface
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(wx, wy, z_pred, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z (Predicted)')
# ax.set_title('Kriging Prediction Surface')

# # Add a color bar which maps values to colors
# fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.view_init(elev=35, azim=-75)

# plt.savefig('kriging_pred_surface_prescaled_data.png')
# plt.show()


# # Plot the 3D kriging residuals
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z variance')
# ax.set_title('Kriging Variance Surface')

# # Add a color bar which maps values to colors
# fig.colorbar(surf, shrink=0.5, aspect=5)

# ax.view_init(elev=35, azim=-60)

# plt.savefig('kriging_var_surface_prescaled_data.png')
# plt.show()



#%% Kriging prediction surface stanardscaled data


from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Reshape z to a 2D array as StandardScaler expects a 2D array as input
z_2d = z.reshape(-1, 1)

# Fit the scaler to the data and transform the data
z_standardized = scaler.fit_transform(z_2d)

# Reshape the standardized data back to a 1D array
z_std = z_standardized.flatten()

# Variogram with custom parameters
variogram_parameters = {'range': 2364.711, 'sill': 1.029, 'nugget': 13e-2}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z_std,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred_scl, ss = OK.execute('grid', wxvec, wyvec)

vari = np.mean(abs(ss))
print('Kriging Variance standardscaled: {}'.format(vari))

# Assuming you have already calculated z_pred and created wxvec, wyvec

# Create meshgrid from wxvec and wyvec
wx, wy = np.meshgrid(wxvec, wyvec)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, z_pred_scl, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('Kriging Prediction Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_pred_surface_standardscaled.png')
plt.show()


# Plot the 3D kriging residuals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(wx, wy, ss, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z variance')
ax.set_title('Kriging Variance Surface')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=35, azim=-60)

plt.savefig('kriging_var_surface_standardscaled.png')
plt.show()


#%% DEM convolution


### Plot DEM convolution surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(wx, wy, DEM_convolution, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('DEM convolution surface')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=40, azim=-55)
plt.savefig('DEM_convolution_surface.png')
# ax.view_init(elev=45, azim=-120)


plt.show()


#%% Kriging absolute surface

Z_kriging_absolute = DEM_convolution - z_pred

# Plot the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the predicted plane
surf = ax.plot_surface(wx, wy, Z_kriging_absolute, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Predicted)')
ax.set_title('KNN absolute pred surface')

# Add color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(elev=40, azim=-55)
plt.savefig('kriging_abs_pred_surface.png')
# ax.view_init(elev=45, azim=-120)


plt.show()




