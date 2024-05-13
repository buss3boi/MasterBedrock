# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:24:53 2024

@author: busse
"""

import pandas as pd
import os
import sys


script_path = os.path.abspath(sys.argv[0])

# Add the granada folder to the Python path
DEM_path = os.path.join(os.path.dirname(script_path), '..', 'DEM')
sys.path.append(DEM_path)

DEM_df_path = os.path.join(DEM_path, 'DEM_subset.csv')

# Read OBS_XYZ.csv into a DataFrame
DEM_df = pd.read_csv(DEM_df_path)




#%% DEM convolution surface


import pandas as pd
import numpy as np



def moving_window_average(df, window_shape, stride):
    """Calculates moving average over a DataFrame with a given window size and stride.

    Args:
        df: The input DataFrame.
        window_shape: A tuple (window_height, window_width).
        stride: A tuple (stride_y, stride_x).

    Returns:
        A new DataFrame with the averaged values.
    """

    new_rows = int((df.shape[0] - window_shape[0]) / stride[0]) + 1
    new_cols = int((df.shape[1] - window_shape[1]) / stride[1]) + 1
    result_df = pd.DataFrame(np.zeros((new_rows, new_cols)))

    for i in range(new_rows):
        y_start = round(i * stride[0])
        y_end = round(y_start + window_shape[0])
        for j in range(new_cols):
            x_start = round(j * stride[1])
            x_end = round(x_start + window_shape[1])
            window = df.iloc[y_start:y_end, x_start:x_end]
            result_df.iloc[i, j] = window.mean().mean()

    return result_df


# ML models produce prediction surface in 100x100 grid

window_shape = (8.6, 13.9)  # Example: Rectangular window 
stride = (8.6, 13.8999)        # Example: Stride of 3 in y-direction, 2 in x-direction

averaged_df = moving_window_average(DEM_df, window_shape, stride)

averaged_df.to_csv('dem_convoluted.csv', index=False)

# ML models produce prediction surface in 87x140 grid

window_shape = (10, 10)  # Example: Rectangular window 
stride = (9.85, 9.9)        # Example: Stride of 3 in y-direction, 2 in x-direction

averaged_df_kriging = moving_window_average(DEM_df, window_shape, stride)

averaged_df_kriging.to_csv('dem_convoluted_kriging.csv', index=False)
