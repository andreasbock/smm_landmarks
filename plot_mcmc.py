import theano
from theano import function
import theano.tensor as T
from termcolor import colored
from scipy.optimize import minimize, fmin_bfgs, fmin_cg
import numpy as np
import pickle
import sys as sys

from lib import *

args = sys.argv
if len(args) < 2:
    print("No path specified!")
    exit(1)

log_dir = str(args[1])
log_dir += '/'

#test_name = str(args[2])

# compute bounding box for modulus
x_min  = -1.
y_min  = -1.
x_max = 1.
y_max = 1.

# load results
po = open(log_dir + "fnls.pickle", "rb")
fnls = pickle.load( po)
po.close()

po = open(log_dir + "c_samples.pickle", "rb")
c_samples = pickle.load(po)
po.close()



# plotting
#centroid_heatmap(c_samples, log_dir, x_min, x_max, y_min, y_max,bins=15)
#centroid_plot(c_samples, log_dir, x_min, x_max, y_min, y_max)

#plot_autocorr(c_samples, log_dir, lag_max = 1000)
#fnl_histogram(fnls, log_dir,test_name = '', bins=50)

#below does not work, not enough info is saved in the pickle!

# horizontal initial momentum
num_landmarks=10
q0,q1,test_name = squeeze(num_landmarks)
N = theano.shared(len(q0))
p0 = 1./N.eval()*np.vstack((np.zeros(N.eval()), np.ones(N.eval()))).T
p0 = np.array(int(N.get_value())*[[0.1, 0]])

# create the initial condition array
q0 = q0.astype(theano.config.floatX)
p0 = p0.astype(theano.config.floatX)
x0 = np.array([q0, p0]).astype(theano.config.floatX)

for j in range(3):
    j+=1
    name = 'MAP_center_{}'.format(j)
    po = open(log_dir + name + ".pickle", "rb")
    print(po)
    xs = pickle.load(po)
    po.close()
    plot_q(x0, xs, num_landmarks, log_dir + name, nus=centers)#title=str(val))


