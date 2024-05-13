# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:02:37 2023

@author: busse
"""

#%% Plotting 

# Link to plotting: https://betterprogramming.pub/creating-topographic-maps-in-python-convergence-of-art-and-data-science-7492b8c9fa6e

import rasterio
import pandas as pd
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np

src = rasterio.open('6602_2_10m_z33.dem')
dataset = src.read()

print(dataset.shape)

from rasterio.plot import show
show(src, cmap='terrain')

geotransform = src.transform

wminx, wmaxx, wminy, wmaxy = 254100, 268000, 6620100, 6628700

window = rasterio.windows.from_bounds(wminx, wminy, wmaxx, wmaxy, geotransform)
   
# Read the data from the window
subset = src.read(1, window=window)
show(subset, cmap='terrain')


#%% Save DEM subset
window_ul_x, window_ul_y = window.col_off, window.row_off
window_lr_x = window_ul_x + window.width
window_lr_y = window_ul_y + window.height

# Apply geotransform to get real-world coordinates of the window corners
ul_x, ul_y = geotransform * (window_ul_x, window_ul_y)  # Upper left
lr_x, lr_y = geotransform * (window_lr_x, window_lr_y)  # Lower right

# Print the X and Y coordinates (optional)
print(f"Upper Left  (X, Y): ({ul_x:.2f}, {ul_y:.2f})")
print(f"Lower Right (X, Y): ({lr_x:.2f}, {lr_y:.2f})") 

window_width = int(window.width)
window_height = int(window.height)
x_coords = np.linspace(wminx, wmaxx-10, window_width)

# Coordinates along the y-direction (note: reversed due to image indexing)
y_coords = np.linspace(wmaxy, wminy+10, window_height)

# Create DataFrame with coordinates as index and columns
df = pd.DataFrame(subset, index=y_coords, columns=x_coords)


# Save the DataFrame as a CSV file
df.to_csv('DEM_subset.csv', index=False) 


#%% Show DEM with wells


import os
import sys

# Get the current script's path

script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DQM_path = os.path.join(os.path.dirname(script_path), '..', 'Regression')
sys.path.append(DQM_path)

obs_xyz_path = os.path.join(DQM_path, 'OBS_XYZ_gdf.csv')

# Read OBS_XYZ.csv into a DataFrame
obs_xyz_gdf = pd.read_csv(obs_xyz_path)



# Create a single figure and axis for both the image and the red dots
fig, ax = plt.subplots(figsize=(12, 10))

# Show the subset of the DEM data with cmap 'terrain' on the same axis
retted = show(subset, cmap='terrain', ax=ax)
im = retted.get_images()[0]
fig.colorbar(im,fraction=0.029, pad=0.04).outline.set_visible(False)
plt.title('Digital Elevation Model')
plt.box(False)
ax.grid(color='white', linestyle='-', linewidth=0.3, alpha=1) 

# Assuming 'df' has columns 'X', 'Y', and optionally 'Z' for borehole coordinates
x = obs_xyz_gdf['X'].values 
y = obs_xyz_gdf['Y'].values
z = obs_xyz_gdf['Z'].values  # If you have Z values (depths), uncomment this line

# Scale borehole coordinates to match the DEM plot
x_scaled = (x - wminx) / (wmaxx - wminx) * subset.shape[1] 
y_scaled = (wmaxy - y) / (wmaxy - wminy) * subset.shape[0]  # Invert y-axis as DEM plots are usually top-down

# Plot boreholes on the existing plot
ax.scatter(x_scaled, y_scaled, color='red', marker='o', s=5, label='Boreholes')

# Customize the legend
plt.legend(loc='upper right')

plt.savefig('Digital_Elevation_Model.png',  dpi=300, bbox_inches='tight')




