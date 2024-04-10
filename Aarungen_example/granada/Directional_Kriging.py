# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:24:58 2024

@author: busse
"""

from Granada_plotting import OBS_XYZ, wxvec, wyvec

#%% Kriging


# For the SILL, the relation number between most models seems to be roughly 
# East-West / North-South = 0.58, or 1.74 inverse

import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pykrige.kriging_tools as kt

# Extracting X, Y, and Z values from OBS_XYZ
x = OBS_XYZ['Shape_X'].values
y = OBS_XYZ['Shape_Y'].values
z = OBS_XYZ['blengdber_'].values


#%% North-South Spherical kriging

# Variogram Parameters
sill = 1111
nugget = 106
range_param = 722

# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='spherical',  # You can choose a different model if needed
    verbose=False,
    enable_plotting=False,
    variogram_parameters=variogram_parameters
)

# Perform the kriging interpolation
z_pred, ss = OK.execute('grid', wxvec, wyvec)

MSE = np.sqrt(np.mean(ss**2))
print(MSE)

proportion = len(wxvec)/len(wyvec)


# Plot the kriging result
plt.figure(figsize=(8*proportion, 8))
plt.scatter(x, y, c=z, cmap="viridis", marker="o", s=30, label="Borehole Data")
plt.contourf(wxvec, wyvec, z_pred, levels=100, cmap="viridis", alpha=0.8)
plt.colorbar(label="Interpolated Value")
plt.title("North-South Spherical kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

# RMSE on ss: 915


#%% East-West Spherical kriging

# Variogram Parameters
sill = 653
nugget = 26
range_param = 425

# Variogram with custom parameters
variogram_parameters = {'sill': sill, 'nugget': nugget, 'range': range_param}


# Create the kriging object
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model='spherical',  # You can choose a different model if needed
    verbose=False,
    enable_plotting=False,
    variogram_parameters=variogram_parameters
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
plt.title("East-West Spherical kriging")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend(loc='upper left')
plt.show()

# RMSE on ss: 592

# From seing the Kriging Plot, i consider the east-west to be a bad model


# UPDOWN first, SIDEWAYS 2, NOEA-SOWE 3, NOWE-SOEA 4
# Direction Eas-West i always the shortest axis, Northeast - southwest is the longest axis
# Divide the major axis on the ration of the major/minor axis

# tolerance=60, maxlag=1000, n_lags=20

# To adjust data for anisotropy, 
# Model: spherical    range: 721.96   sill: 1174.0   nugget: 98.55
# Model: exponential    range: 872.17   sill: 1451.0   nugget: -93.23
# Model: gaussian    range: 694.26   sill: 1016.2   nugget: 254.54
# Model: spherical    range: 442.49   sill: 680.9   nugget: 31.51
# Model: exponential    range: 632.15   sill: 801.1   nugget: -30.72
# Model: gaussian    range: 429.45   sill: 586.2   nugget: 126.55
# Model: spherical    range: 824.71   sill: 1451.9   nugget: -2.89
# Model: exponential    range: 1278.71   sill: 1831.4   nugget: -130.37
# Model: gaussian    range: 817.01   sill: 1260.9   nugget: 199.65
# Model: spherical    range: 3881171.63   sill: 2873701.8   nugget: 243.09
# Model: exponential    range: 19888391.13   sill: 7363395.0   nugget: 243.08
# Model: gaussian    range: 3662.24   sill: 3883.3   nugget: 446.94


# Trial 2
# Model: spherical    range: 737.74   sill: 1085.6   nugget: 98.67
# Model: exponential    range: 921.58   sill: 1331.5   nugget: -58.79
# Model: gaussian    range: 728.22   sill: 934.1   nugget: 254.55
# Model: spherical    range: 475.46   sill: 642.9   nugget: 48.10
# Model: exponential    range: 642.67   sill: 770.4   nugget: -27.87
# Model: gaussian    range: 463.47   sill: 553.3   nugget: 138.43
# Model: spherical    range: 893.87   sill: 1441.1   nugget: -8.93
# Model: exponential    range: 1632.14   sill: 1905.1   nugget: -88.35
# Model: gaussian    range: 904.47   sill: 1258.7   nugget: 198.30
# Model: spherical    range: 3661871.54   sill: 2539630.3   nugget: 224.90
# Model: exponential    range: 14085983.91   sill: 4885087.4   nugget: 224.88
# Model: gaussian    range: 2059.68   sill: 1492.2   nugget: 386.65

# Trial 3

# Model: spherical    range: 703.13   sill: 1165.2   nugget: 79.82
# Model: exponential    range: 853.24   sill: 1426.8   nugget: -100.25
# Model: gaussian    range: 688.89   sill: 1004.4   nugget: 243.50
# Model: spherical    range: 501.69   sill: 665.8   nugget: 63.11
# Model: exponential    range: 789.23   sill: 794.9   nugget: 18.21
# Model: gaussian    range: 490.88   sill: 573.7   nugget: 156.59
# Model: spherical    range: 817.75   sill: 1423.9   nugget: -15.01
# Model: exponential    range: 1275.83   sill: 1788.0   nugget: -132.36
# Model: gaussian    range: 820.10   sill: 1235.4   nugget: 189.79
# Model: spherical    range: 4413973.39   sill: 3319576.9   nugget: 232.65
# Model: exponential    range: 16362786.85   sill: 6153460.7   nugget: 232.64
# Model: gaussian    range: 1928.52   sill: 1493.8   nugget: 400.92


# Trial 4

# Model: spherical    range: 700.55   sill: 1026.7   nugget: 128.80
# Model: exponential    range: 799.74   sill: 1279.1   nugget: -65.44
# Model: gaussian    range: 675.30   sill: 885.5   nugget: 268.31
# Model: spherical    range: 836.27   sill: 631.6   nugget: 171.18
# Model: exponential    range: 853.51   sill: 816.7   nugget: 9.68
# Model: gaussian    range: 549.23   sill: 572.9   nugget: 164.69
# Model: spherical    range: 960.36   sill: 1462.2   nugget: 26.57
# Model: exponential    range: 1800.88   sill: 1992.6   nugget: -64.16
# Model: gaussian    range: 938.30   sill: 1274.3   nugget: 220.89
# Model: spherical    range: 4703943.87   sill: 3132927.3   nugget: 252.10
# Model: exponential    range: 50003.99   sill: 17183.0   nugget: 246.30
# Model: gaussian    range: 1715.87   sill: 1154.0   nugget: 391.32


# Trial 5
# Model: spherical    range: 720.40   sill: 1155.3   nugget: 93.26
# Model: exponential    range: 898.17   sill: 1422.9   nugget: -80.51
# Model: gaussian    range: 704.94   sill: 996.5   nugget: 254.69
# Model: spherical    range: 877.44   sill: 657.0   nugget: 183.46
# Model: exponential    range: 933.60   sill: 841.2   nugget: 30.38
# Model: gaussian    range: 571.77   sill: 583.9   nugget: 180.71
# Model: spherical    range: 815.53   sill: 1448.9   nugget: -23.85
# Model: exponential    range: 1282.03   sill: 1819.9   nugget: -139.80
# Model: gaussian    range: 819.05   sill: 1257.7   nugget: 184.76
# Model: spherical    range: 3622021.22   sill: 2657290.3   nugget: 230.78
# Model: exponential    range: 17814615.40   sill: 6535421.8   nugget: 230.76
# Model: gaussian    range: 2469.36   sill: 2027.2   nugget: 415.77


# Trial 6
# Model: spherical    range: 726.97   sill: 1080.8   nugget: 87.92
# Model: exponential    range: 931.85   sill: 1320.9   nugget: -57.18
# Model: gaussian    range: 722.35   sill: 930.2   nugget: 244.95
# Model: spherical    range: 903.58   sill: 610.1   nugget: 184.54
# Model: exponential    range: 1013.03   sill: 776.7   nugget: 56.54
# Model: gaussian    range: 638.36   sill: 525.2   nugget: 205.53
# Model: spherical    range: 919.49   sill: 1439.8   nugget: -2.76
# Model: exponential    range: 1760.13   sill: 1943.3   nugget: -73.61
# Model: gaussian    range: 930.25   sill: 1259.8   nugget: 203.13
# Model: spherical    range: 2775281.42   sill: 1795143.7   nugget: 243.29
# Model: exponential    range: 45044235.38   sill: 14568587.3   nugget: 243.28
# Model: gaussian    range: 2244.90   sill: 1557.9   nugget: 401.05


# Trial 7
# Model: spherical    range: 722.97   sill: 1131.3   nugget: 84.51
# Model: exponential    range: 877.95   sill: 1388.7   nugget: -91.35
# Model: gaussian    range: 709.74   sill: 973.6   nugget: 245.14
# Model: spherical    range: 449.02   sill: 682.5   nugget: 14.37
# Model: exponential    range: 576.34   sill: 815.6   nugget: -76.10
# Model: gaussian    range: 442.28   sill: 584.4   nugget: 114.03
# Model: spherical    range: 848.40   sill: 1476.3   nugget: -23.78
# Model: exponential    range: 1404.95   sill: 1886.4   nugget: -129.05
# Model: gaussian    range: 853.82   sill: 1284.0   nugget: 188.55
# Model: spherical    range: 3400355.48   sill: 2244420.0   nugget: 240.07
# Model: exponential    range: 14372843.58   sill: 4743926.8   nugget: 240.06
# Model: gaussian    range: 2809.01   sill: 2218.2   nugget: 413.46


# Model: spherical    range: 714.31   sill: 1166.5   nugget: 97.56
# Model: exponential    range: 858.10   sill: 1433.6   nugget: -89.47
# Model: gaussian    range: 699.45   sill: 1004.8   nugget: 261.68
# Model: spherical    range: 499.13   sill: 680.5   nugget: 36.22
# Model: exponential    range: 738.12   sill: 811.9   nugget: -22.01
# Model: gaussian    range: 494.32   sill: 584.0   nugget: 135.38
# Model: spherical    range: 832.21   sill: 1508.8   nugget: -21.79
# Model: exponential    range: 1328.17   sill: 1907.9   nugget: -140.06
# Model: gaussian    range: 834.61   sill: 1310.9   nugget: 194.22
# Model: spherical    range: 2906768.00   sill: 2047479.7   nugget: 247.94
# Model: exponential    range: 9152164.31   sill: 3223867.9   nugget: 247.90
# Model: gaussian    range: 1991.25   sill: 1445.3   nugget: 411.13


# Model: spherical    range: 695.18   sill: 1119.0   nugget: 112.53
# Model: exponential    range: 830.44   sill: 1375.0   nugget: -69.83
# Model: gaussian    range: 681.81   sill: 962.7   nugget: 271.35
# Model: spherical    range: 503.77   sill: 688.1   nugget: 66.56
# Model: exponential    range: 805.93   sill: 824.7   nugget: 21.88
# Model: gaussian    range: 481.03   sill: 597.8   nugget: 155.81
# Model: spherical    range: 840.10   sill: 1457.1   nugget: -1.27
# Model: exponential    range: 1368.46   sill: 1851.0   nugget: -108.64
# Model: gaussian    range: 848.99   sill: 1265.8   nugget: 211.12
# Model: spherical    range: 3994318.97   sill: 3036430.1   nugget: 250.71
# Model: exponential    range: 12421376.04   sill: 4721875.5   nugget: 250.69
# Model: gaussian    range: 2556.94   sill: 2217.8   nugget: 443.62


# Model: spherical    range: 715.17   sill: 1135.3   nugget: 76.00
# Model: exponential    range: 878.65   sill: 1394.0   nugget: -97.21
# Model: gaussian    range: 700.41   sill: 978.5   nugget: 235.43
# Model: spherical    range: 442.19   sill: 659.1   nugget: 22.87
# Model: exponential    range: 573.78   sill: 787.6   nugget: -63.14
# Model: gaussian    range: 434.71   sill: 564.4   nugget: 118.86
# Model: spherical    range: 817.22   sill: 1481.9   nugget: -44.80
# Model: exponential    range: 1264.71   sill: 1854.7   nugget: -167.55
# Model: gaussian    range: 820.30   sill: 1285.8   nugget: 168.68
# Model: spherical    range: 2821129.42   sill: 1879744.3   nugget: 238.36
# Model: exponential    range: 19181164.44   sill: 6390799.1   nugget: 238.35
# Model: gaussian    range: 2975.22   sill: 2457.1   nugget: 415.56





# UPDOWN first, SIDEWAYS 2, NOEA-SOWE 3, NOWE-SOEA 4
# Direction Eas-West i always the shortest axis, Northeast - southwest is the longest axis
# Divide the major axis on the ration of the major/minor axis

# tolerance=60, maxlag=0.5, n_lags=20


# Model: spherical    range: 839.13   sill: 1191.5   nugget: 134.86
# Model: exponential    range: 1601.92   sill: 1660.2   nugget: 55.09
# Model: gaussian    range: 814.43   sill: 1030.1   nugget: 294.17
# Model: spherical    range: 416.93   sill: 661.7   nugget: 17.00
# Model: exponential    range: 535.65   sill: 792.5   nugget: -71.40
# Model: gaussian    range: 406.01   sill: 568.5   nugget: 110.65
# Model: spherical    range: 845.86   sill: 1429.7   nugget: 3.75
# Model: exponential    range: 1939.40   sill: 2138.5   nugget: -48.11
# Model: gaussian    range: 838.26   sill: 1249.3   nugget: 198.35
# Model: spherical    range: 793.91   sill: 672.9   nugget: 223.79
# Model: exponential    range: 860.02   sill: 863.8   nugget: 69.92
# Model: gaussian    range: 579.36   sill: 588.2   nugget: 249.40


# Model: spherical    range: 778.52   sill: 1209.0   nugget: 96.35
# Model: exponential    range: 1436.68   sill: 1642.6   nugget: 16.45
# Model: gaussian    range: 773.29   sill: 1052.0   nugget: 265.10
# Model: spherical    range: 424.62   sill: 658.2   nugget: 12.45
# Model: exponential    range: 561.33   sill: 784.9   nugget: -67.03
# Model: gaussian    range: 413.13   sill: 566.7   nugget: 104.51
# Model: spherical    range: 895.54   sill: 1516.0   nugget: -3.05
# Model: exponential    range: 2331.22   sill: 2441.7   nugget: -41.87
# Model: gaussian    range: 876.98   sill: 1318.9   nugget: 198.17
# Model: spherical    range: 858.53   sill: 736.5   nugget: 203.20
# Model: exponential    range: 1178.33   sill: 948.1   nugget: 97.97
# Model: gaussian    range: 723.95   sill: 619.0   nugget: 275.20


# Model: spherical    range: 788.82   sill: 1189.3   nugget: 88.99
# Model: exponential    range: 1511.34   sill: 1636.4   nugget: 18.64
# Model: gaussian    range: 783.47   sill: 1036.5   nugget: 253.93
# Model: spherical    range: 415.14   sill: 634.7   nugget: 24.39
# Model: exponential    range: 556.37   sill: 755.1   nugget: -49.72
# Model: gaussian    range: 403.53   sill: 546.3   nugget: 113.17
# Model: spherical    range: 867.77   sill: 1385.3   nugget: 11.30
# Model: exponential    range: 2130.66   sill: 2150.2   nugget: -29.35
# Model: gaussian    range: 859.19   sill: 1210.6   nugget: 198.90
# Model: spherical    range: 1087.92   sill: 823.3   nugget: 228.62
# Model: exponential    range: 1587.24   sill: 1031.6   nugget: 140.97
# Model: gaussian    range: 859.28   sill: 635.7   nugget: 309.84


# Model: spherical    range: 782.58   sill: 1150.5   nugget: 125.94
# Model: exponential    range: 1351.33   sill: 1537.9   nugget: 32.27
# Model: gaussian    range: 761.58   sill: 997.0   nugget: 280.61
# Model: spherical    range: 956.21   sill: 679.9   nugget: 188.92
# Model: exponential    range: 1057.39   sill: 833.7   nugget: 59.71
# Model: gaussian    range: 588.75   sill: 556.3   nugget: 194.24
# Model: spherical    range: 861.00   sill: 1414.9   nugget: 14.14
# Model: exponential    range: 2052.86   sill: 2163.1   nugget: -32.10
# Model: gaussian    range: 854.96   sill: 1236.4   nugget: 207.54
# Model: spherical    range: 928.21   sill: 750.7   nugget: 223.33
# Model: exponential    range: 1261.07   sill: 955.1   nugget: 118.73
# Model: gaussian    range: 765.28   sill: 615.2   nugget: 296.73


# Model: spherical    range: 752.78   sill: 1228.2   nugget: 82.80
# Model: exponential    range: 1327.63   sill: 1630.0   nugget: -2.68
# Model: gaussian    range: 760.79   sill: 1068.6   nugget: 261.76
# Model: spherical    range: 438.63   sill: 652.7   nugget: 33.51
# Model: exponential    range: 693.20   sill: 767.1   nugget: -3.28
# Model: gaussian    range: 431.93   sill: 559.6   nugget: 128.21
# Model: spherical    range: 883.12   sill: 1470.6   nugget: 0.58
# Model: exponential    range: 2243.73   sill: 2329.5   nugget: -38.53
# Model: gaussian    range: 874.14   sill: 1284.5   nugget: 199.65
# Model: spherical    range: 983.35   sill: 833.7   nugget: 220.81
# Model: exponential    range: 1531.63   sill: 1088.3   nugget: 133.32
# Model: gaussian    range: 850.20   sill: 679.4   nugget: 315.18


# Model: spherical    range: 800.83   sill: 1142.4   nugget: 133.00
# Model: exponential    range: 1417.58   sill: 1544.3   nugget: 43.35
# Model: gaussian    range: 776.98   sill: 988.5   nugget: 285.88
# Model: spherical    range: 425.23   sill: 660.7   nugget: 14.18
# Model: exponential    range: 564.26   sill: 787.3   nugget: -64.53
# Model: gaussian    range: 415.09   sill: 568.3   nugget: 107.50
# Model: spherical    range: 917.68   sill: 1391.3   nugget: 42.61
# Model: exponential    range: 2231.49   sill: 2174.0   nugget: -7.24
# Model: gaussian    range: 874.47   sill: 1195.6   nugget: 220.04
# Model: spherical    range: 834.33   sill: 739.5   nugget: 202.52
# Model: exponential    range: 1082.63   sill: 947.4   nugget: 83.83
# Model: gaussian    range: 699.27   sill: 625.5   nugget: 272.42


# Model: spherical    range: 769.82   sill: 1218.5   nugget: 88.03
# Model: exponential    range: 1384.48   sill: 1638.1   nugget: 3.04
# Model: gaussian    range: 760.08   sill: 1059.7   nugget: 255.89
# Model: spherical    range: 425.41   sill: 658.8   nugget: 15.37
# Model: exponential    range: 553.54   sill: 788.2   nugget: -68.96
# Model: gaussian    range: 417.64   sill: 564.9   nugget: 110.55
# Model: spherical    range: 879.30   sill: 1490.3   nugget: -6.95
# Model: exponential    range: 2200.54   sill: 2342.9   nugget: -49.40
# Model: gaussian    range: 861.67   sill: 1298.4   nugget: 190.83
# Model: spherical    range: 826.12   sill: 775.7   nugget: 188.26
# Model: exponential    range: 1103.25   sill: 997.7   nugget: 69.89
# Model: gaussian    range: 683.98   sill: 660.2   nugget: 256.43

# Model: spherical    range: 709.76   sill: 1130.2   nugget: 103.17
# Model: exponential    range: 1081.55   sill: 1445.7   nugget: -12.09
# Model: gaussian    range: 694.25   sill: 979.0   nugget: 257.89
# Model: spherical    range: 431.50   sill: 648.6   nugget: 22.50
# Model: exponential    range: 570.72   sill: 775.8   nugget: -57.18
# Model: gaussian    range: 420.34   sill: 557.5   nugget: 114.13
# Model: spherical    range: 781.47   sill: 1350.9   nugget: 2.43
# Model: exponential    range: 1568.48   sill: 1879.8   nugget: -62.59
# Model: gaussian    range: 788.09   sill: 1181.7   nugget: 194.72
# Model: spherical    range: 851.78   sill: 772.1   nugget: 196.41
# Model: exponential    range: 1206.35   sill: 999.8   nugget: 92.16
# Model: gaussian    range: 737.12   sill: 652.1   nugget: 277.24


# Model: spherical    range: 738.40   sill: 1144.4   nugget: 102.78
# Model: exponential    range: 1219.64   sill: 1496.9   nugget: 4.97
# Model: gaussian    range: 729.07   sill: 992.9   nugget: 262.02
# Model: spherical    range: 459.20   sill: 673.0   nugget: 24.99
# Model: exponential    range: 679.11   sill: 803.1   nugget: -32.01
# Model: gaussian    range: 448.49   sill: 579.9   nugget: 119.20
# Model: spherical    range: 829.71   sill: 1445.4   nugget: -8.69
# Model: exponential    range: 1907.30   sill: 2158.0   nugget: -58.49
# Model: gaussian    range: 830.28   sill: 1266.1   nugget: 191.75
# Model: spherical    range: 694.85   sill: 714.8   nugget: 171.54
# Model: exponential    range: 888.37   sill: 915.6   nugget: 50.72
# Model: gaussian    range: 604.70   sill: 623.6   nugget: 240.82


# Model: spherical    range: 756.93   sill: 1127.8   nugget: 113.85
# Model: exponential    range: 1262.73   sill: 1485.8   nugget: 16.76
# Model: gaussian    range: 740.43   sill: 978.0   nugget: 267.32
# Model: spherical    range: 411.85   sill: 655.8   nugget: 9.39
# Model: exponential    range: 528.30   sill: 785.0   nugget: -78.55
# Model: gaussian    range: 397.89   sill: 565.4   nugget: 99.63
# Model: spherical    range: 803.21   sill: 1381.0   nugget: -0.63
# Model: exponential    range: 1672.00   sill: 1962.2   nugget: -63.80
# Model: gaussian    range: 797.27   sill: 1207.1   nugget: 188.38
# Model: spherical    range: 757.36   sill: 692.5   nugget: 195.33
# Model: exponential    range: 890.43   sill: 889.9   nugget: 56.67
# Model: gaussian    range: 603.86   sill: 603.6   nugget: 242.69