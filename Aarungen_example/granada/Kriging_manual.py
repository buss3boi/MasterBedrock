# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:01:54 2024

@author: busse
"""
import pandas as pd
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from pykrige.ok import OrdinaryKriging

from Granada_plotting import wxvec, wyvec, dx, dy

# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'DQM')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
OBS_XYZ_gdf = pd.read_csv(obs_xyz_path)



#%% Anisotropic scaling from DEM file results

# Anisotropic scaling is done via the rotation matrix. Parameters are found 
# from the scaled values

# 10 trials. Compute mean of different Directional Models

minor_axis = {'spherical':[416.93, 424.62, 415.14, 956.21, 438.63, 425.23, 425.41, 431.5, 459.2, 411.85], 
              'exponential':[535.65, 561.33, 556.37, 1057.39, 693.2, 564.26, 553.54, 570.72, 679.11, 528.30], 
              'gaussian':[406.01, 413.13, 403.53, 588.75, 431.93, 415.09, 417.64, 420.34, 448.49, 397.89]}


major_axis = {'spherical':[845.86, 895.54, 867.77, 861, 883.12, 917.68, 879.3, 781.47, 829.71, 803.21],
              'exponential':[1939.4, 2331.22, 2130.66, 2052.86, 2243.73, 2231.49, 2200.54, 1568.48, 1907.3, 1672],
              'gaussian':[838.26, 876.98, 859.19, 854.96, 874.14, 874.47, 861.67, 788.09, 830.28, 797.27]}


# Calculate the ratios
ratios = {}
for model in minor_axis.keys():
    ratios[model] = [major / minor for major, minor in zip(major_axis[model], minor_axis[model])]


# Assuming you already have the 'ratios' dictionary from the previous code snippet

# Calculate the median for each model
medians = {model: np.median(values) for model, values in ratios.items()}

# Print the resulting medians
for model, median in medians.items():
    print(f'{model} median ratio:', median)
    


#%% Apply aspect ratio with the rotation matrix


# Step 1: Rotate the Dataset
rotation_angle = 45  # Example rotation angle in degrees
rotation_angle_rad = np.radians(rotation_angle)



def rotation_matrix(matrix, theta):
    """
    Create a rotation matrix for the given angle theta (in degrees) applied to the input matrix.
    """
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s],
                                [s, c]])
    return np.dot(matrix, rotation_matrix)


rotated_data = rotation_matrix(OBS_XYZ_gdf[['Shape_X', 'Shape_Y']], rotation_angle)


scaling_factor = 2.2  # Example scaling factor
scaled_data = rotated_data.copy()
scaled_data[:, 1] /= scaling_factor


final_data = rotation_matrix(scaled_data, -25)




plt.figure(figsize=(10, 6))

# Plot original data
plt.scatter(OBS_XYZ_gdf['Shape_X'], OBS_XYZ_gdf['Shape_Y'], label='Original Data', marker='o', color='blue')

plt.figure(figsize=(10, 6))
# Plot rotated data
plt.scatter(rotated_data[:, 0], rotated_data[:, 1], label='Rotated Data', marker='x', color='orange')

plt.figure(figsize=(10, 6))
# Plot scaled data
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], label='Scaled Data', marker='s', color='green')

plt.figure(figsize=(10, 6))
# Plot final data (rotated back)
plt.scatter(final_data[:, 0], final_data[:, 1], label='Final Data (Rotated Back)', marker='^', color='red')

# Set labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original, Rotated, Scaled, and Final Data')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



#%% Borrow function from variogram_anisotropic
from scipy.optimize import curve_fit
from skgstat import models

def fit_and_plot_variogram(variogram, model_func, initial_guess=None):
    xdata = variogram.bins
    ydata = variogram.experimental

    # Initial guess - if not provided, use default values
    if initial_guess is None:
        initial_guess = [np.mean(xdata), np.mean(ydata), 0]

    # Fit the model to the variogram data
    coef, cov = curve_fit(model_func, xdata, ydata, p0=initial_guess)

    # Print model parameters
    print('Model: {}    range: {:.2f}   sill: {:.1f}   nugget: {:.2f}'.format(model_func.__name__, coef[0], coef[1], coef[2]))

    # Generate y values for the fitted model
    xi = np.linspace(xdata[0], xdata[-1], 100)
    yi = [model_func(h, *coef) for h in xi]

    # Plot the variogram data and the fitted model
    # plt.plot(xdata, ydata, 'og', label='Experimental')
    # plt.plot(xi, yi, '-b', label='Fitted Model')

    # # Set plot title indicating directional variogram and model
    # plt.title(f'{variogram}')
    # plt.legend()
    # plt.show()
    
    return xi, yi, coef


#%% Fit curves to experimental semivariograms

# 1
simple_variogram = skg.Variogram(OBS_XYZ_gdf[['Shape_X', 'Shape_Y']], OBS_XYZ_gdf['blengdber_'], n_lags=100)
simple_semivar = fit_and_plot_variogram(simple_variogram, models.exponential)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(simple_variogram.bins, simple_variogram.experimental, '.--r', label='Anisotropic Variogram')
ax.plot(simple_semivar[0], simple_semivar[1], '-r', label='Anisotropic semivar')

ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')
plt.savefig('skgstat_semivariogram.png')

# 2
isotropic_variogram = skg.Variogram(final_data, OBS_XYZ_gdf['blengdber_'], n_lags=100)
isotropic_semivar = fit_and_plot_variogram(isotropic_variogram, models.exponential)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(simple_variogram.bins, simple_variogram.experimental, '.--r', label='Anisotropic Variogram')
ax.plot(simple_semivar[0], simple_semivar[1], '-r', label='Anisotropic semivar')

ax.plot(isotropic_variogram.bins, isotropic_variogram.experimental, '.--b', label='Isotropic Variogram')
ax.plot(isotropic_semivar[0], isotropic_semivar[1], '-b', label='Isotropic semivar')


ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.legend(loc='upper left')
plt.savefig('both_skgstat_semivariograms.png')



#%% See directional variogram with and without finalized data

coordinates = OBS_XYZ_gdf[['Shape_X', 'Shape_Y']]
coord_vals = OBS_XYZ_gdf['blengdber_']

maxlag = 8500
nlags = 50
Vnorth = skg.DirectionalVariogram(coordinates, coord_vals, azimuth=90, tolerance=60, maxlag=maxlag, n_lags=nlags)

Veast = skg.DirectionalVariogram(coordinates, coord_vals, azimuth=0, tolerance=60, maxlag=maxlag, n_lags=nlags)

Vnowe = skg.DirectionalVariogram(coordinates, coord_vals, azimuth=45, tolerance=60, maxlag=maxlag, n_lags=nlags)

Vnoea = skg.DirectionalVariogram(coordinates, coord_vals, azimuth=135, tolerance=60, maxlag=maxlag, n_lags=nlags)



fix, ax = plt.subplots(1,1,figsize=(9,7))

ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
ax.plot(Vnoea.bins, Vnoea.experimental, '.--g', label='Noea-Sowe')
ax.plot(Vnowe.bins, Vnowe.experimental, '.--y', label='Nowe-Soea')


ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.title('Non scaled data')
plt.legend(loc='upper left')
plt.savefig('Directional_variogram.png', dpi=300, bbox_inches='tight')


### Directional Variograms with scaled data

Vnorth = skg.DirectionalVariogram(final_data, coord_vals, azimuth=90, tolerance=60, maxlag=maxlag, n_lags=nlags)

Veast = skg.DirectionalVariogram(final_data, coord_vals, azimuth=0, tolerance=60, maxlag=maxlag, n_lags=nlags)

Vnowe = skg.DirectionalVariogram(final_data, coord_vals, azimuth=45, tolerance=60, maxlag=maxlag, n_lags=nlags)

Vnoea = skg.DirectionalVariogram(final_data, coord_vals, azimuth=135, tolerance=60, maxlag=maxlag, n_lags=nlags)



fix, ax = plt.subplots(1,1,figsize=(9,7))

ax.plot(Vnorth.bins, Vnorth.experimental, '.--r', label='North-South')
ax.plot(Veast.bins, Veast.experimental, '.--b', label='East-West')
ax.plot(Vnoea.bins, Vnoea.experimental, '.--g', label='Noea-Sowe')
ax.plot(Vnowe.bins, Vnowe.experimental, '.--y', label='Nowe-Soea')


ax.set_xlabel('lag [m]')
ax.set_ylabel('semi-variance (matheron)')
plt.title('Scaled data')
plt.legend(loc='upper left')
plt.savefig('FDirectional_variogram.png', dpi=300, bbox_inches='tight')



#%% Compute kriging simple semivariogram

# Extracting X, Y, and Z values from OBS_XYZ
x = OBS_XYZ_gdf['Shape_X'].values
y = OBS_XYZ_gdf['Shape_Y'].values
z = OBS_XYZ_gdf['blengdber_'].values


# Exp curve_fit auto: range: 1826.09   sill: 45.8   nugget: -2.67

range_param = 2275
sill = 1
nugget = 1

# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    #variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)


# We use RMSE of the kriging errors as a quick measure as we belive it is a 
# semi- accurate method
RMSE = np.sqrt(np.mean(ss**2))
print(RMSE)

proportion = len(wxvec)/len(wyvec)
plot_size = 8


# Plot the kriging result
plt.figure(figsize=(plot_size*proportion, plot_size))
plt.scatter(x, y, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(wxvec, wyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("Regular data Exponential kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

### Find results for this model in Kriging_ML


#%% Rotation of data Pykrige rotation


# Exp curve_fit auto: range: 1403.38   sill: 45.2   nugget: 1, maxlag=1
# Exp model auto: range: 5661, sill: 50, nugget: 1.7570e-07
range_param = 1400
sill = 45
nugget = 1

# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    anisotropy_angle=-rotation_angle,
    anisotropy_scaling=scaling_factor,
    #variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

RMSE = np.sqrt(np.mean(ss**2))
print(RMSE)

proportion = len(wxvec)/len(wyvec)
plot_size = 8


# Plot the kriging result
plt.figure(figsize=(plot_size*proportion, plot_size))
plt.scatter(x, y, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(wxvec, wyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("Pykrige scaled Exponential kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

# It is weird because the kriging variogram model, and curve fitted variogram 
# model are drastically different


#%% Manually rotated and scaled data

# Exp model auto: range: 2573, sill: 50, nugget: 2.5333e-07 

# Extracting x and y values from final_data. z values remains the same
fx = final_data[:, 0]
fy = final_data[:, 1]


# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


# Create the kriging object
OK = OrdinaryKriging(
    fx,
    fy,
    z,
    variogram_model='exponential',
    verbose=True,
    enable_plotting=True,
    #variogram_parameters=variogram_parameters
)

### New wec vectors must be created for the final data. We want the same 
# aspect_ratio from dy to fdy. x axis has been stretched so calculate for y, 
# and use the same value fdx fdy

minx, maxx, = OBS_XYZ_gdf['Shape_X'].min(), OBS_XYZ_gdf['Shape_X'].max()
miny, maxy, = OBS_XYZ_gdf['Shape_Y'].min(), OBS_XYZ_gdf['Shape_Y'].max()

aspect_xratio = (maxx-minx)/dx
aspect_yratio = (maxy-miny)/dy


fminx, fmaxx, = final_data[:, 0].min(), final_data[:, 0].max()
fminy, fmaxy, = final_data[:, 1].min(), final_data[:, 1].max()

fdy = fdx = round(dy*(fmaxy-fminy)/(maxy-miny))

fwxvec = np.arange(fminx, fmaxx + fdx, fdx, dtype=np.float64)
fwyvec = np.arange(fminy, fmaxy + fdy, fdy, dtype=np.float64)

# Perform the kriging interpolation with newly computed final_data grid
z_pred, ss = OK.execute('grid', fwxvec, fwyvec)

RMSE = np.sqrt(np.mean(ss**2))
print(RMSE)

proportion = len(fwxvec)/len(fwyvec)
plot_size = 8


# Plot the kriging result
plt.figure(figsize=(plot_size*1.6, plot_size))
plt.scatter(fx, fy, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(fwxvec, fwyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("manually scaled Exponential kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()


#%% Autofit Scaled values


from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Reshape z to a 2D array as StandardScaler expects a 2D array as input
z_2d = z.reshape(-1, 1)

# Fit the scaler to the data and transform the data
z_standardized = scaler.fit_transform(z_2d)

# Reshape the standardized data back to a 1D array
z = z_standardized.flatten()

# Exp model auto: Sill: 1.029, Range: 2364.711, nugget: 10e-10

range_param = 2265
sill = 1.03
nugget = 10e-10


# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


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

# We use RMSE of the kriging errors as a quick measure as we belive it is a 
# semi- accurate method
RMSE = np.sqrt(np.mean(ss**2))
print(RMSE)

proportion = len(wxvec)/len(wyvec)
plot_size = 8


# Plot the kriging result
plt.figure(figsize=(plot_size*proportion, plot_size))
plt.scatter(x, y, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(wxvec, wyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("Standardscaled data Exponential kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

### Find results for this model in Kriging_ML
