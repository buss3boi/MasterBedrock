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


# Create a single figure and axis for both the image and the red dots
fig, ax = plt.subplots(figsize=(10, 8))

# Show the subset of the DEM data with cmap 'terrain' on the same axis
retted = show(subset, cmap='terrain', ax=ax)
im = retted.get_images()[0]
fig.colorbar(im,fraction=0.029, pad=0.04)
plt.title('Digital Elevation Model')
plt.savefig('Digital_Elevation_Model.png',  dpi=300, bbox_inches='tight')



