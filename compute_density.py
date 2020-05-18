from run_custom_nu import run_mm_custom_nu
from lib import *
import numpy as np
import pickle
import os

resolution = 2

# squeeze test
print("computing squeeze test...")
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xs, ys, hs = [], [], []
for x in np.linspace(x_min, x_max, resolution):
    for y in np.linspace(y_min, y_max, resolution):
        nu = [x, y]
        h = run_mm_custom_nu(*squeeze(num_landmarks=10), nus=[nu],
            return_dict=False, plot=False)
        xs.append(x)
        ys.append(y)
        hs.append(h)

po = open("density_squeeze.pickle", "wb")
pickle.dump((xs, ys, hs), po)
po.close()
