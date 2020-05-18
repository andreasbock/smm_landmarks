from run_custom_nu import run_mm_custom_nu
from lib import *
import numpy as np
import pickle
import os

resolution = 20

# criss-cross test
print("computing criss-cross test...")
x_min, x_max = -1, 1
y_min, y_max = -1, 1
xs, ys, hs = [], [], []
for x in np.linspace(x_min, x_max, resolution):
    for y in np.linspace(y_min, y_max, resolution):
        nu = [x, y]
        print(nu)
        h = run_mm_custom_nu(*criss_cross(num_landmarks=10), nus=[nu],
            return_dict=False, plot=False)
        xs.append(x)
        ys.append(y)
        hs.append(h)

po = open("density_criss_cross.pickle", "wb")
pickle.dump((xs, ys, hs), po)
po.close()
